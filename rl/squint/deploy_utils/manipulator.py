"""
Adapted from https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/lerobot/manipulator.py to support SO101.
Code based on https://github.com/huggingface/lerobot for supporting real robot control via the unified LeRobot interface.
"""

import time
from typing import Optional

import numpy as np
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.structs.types import Array

from lerobot.cameras.camera import Camera
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.robots.robot import Robot
from lerobot.utils.robot_utils import precise_sleep


class LeRobotRealAgent(BaseRealAgent):
    """
    LeRobotRealAgent is a general class for controlling real robots via the LeRobot system. You simply just pass in the Robot instance you create via LeRobot and pass it here to make it work with ManiSkill Sim2Real environment interfaces.

    Args:
        robot (Robot): The Robot instance you create via LeRobot.
        use_cached_qpos (bool): Whether to cache the fetched qpos values. If True, the qpos will be
            read from the cache instead of the real robot when possible. This cache is only invalidated when
            set_target_qpos or set_target_qvel is called. This can be useful if you want to easily have higher frequency (> 30Hz) control since qpos reading from the robot is
            currently the slowest part of LeRobot for some of the supported motors.
    """

    def __init__(self, robot: Robot, use_cached_qpos: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._captured_sensor_data = None
        self.real_robot = robot
        self.use_cached_qpos = use_cached_qpos
        self._cached_qpos = None
        self._motor_keys: list[str] = None
        # Fixed 6-motor order expected by the sim (DO NOT iterate bus.motors directly,
        # since wrist_roll may have been removed from bus.motors to skip the handshake check).
        self._all_motor_keys = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        # Motors to skip (hardware issue: wrist_roll has mechanical friction causing overload).
        # Reads return 0.0 (sim thinks joint is at neutral); writes are dropped (motor stays still).
        self._disabled_motors = {"wrist_roll"}
        # Track motors that reported overload during runtime; once flagged, skip future writes
        self._overload_motors: set[str] = set()

        if self.real_robot.name == "so100_follower":
            self.real_robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        elif self.real_robot.name == "so101_follower":
            self.real_robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
            # Gripper mapping from measured values:
            # Sim -10° (closed) <-> Servo -63.82° (closed)
            # Sim 120° (open)   <-> Servo 64.62° (open)
            self._gripper_sim_min = -10.0     # degrees (closed in sim)
            self._gripper_sim_max = 120.0     # degrees (open in sim)
            self._gripper_servo_min = -62.5  # artifical to match sim
            self._gripper_servo_max = 64.62   # degrees (open on real servo)
            self._gripper_sim_range = self._gripper_sim_max - self._gripper_sim_min
            self._gripper_servo_range = self._gripper_servo_max - self._gripper_servo_min

    def start(self):
        self.real_robot.connect()

    def stop(self):
        self.real_robot.disconnect()

    def set_target_qpos(self, qpos: Array):
        self._cached_qpos = None
        qpos = common.to_cpu_tensor(qpos).flatten()
        qpos = torch.rad2deg(qpos)
        qpos = {f"{self._motor_keys[i]}.pos": qpos[i] for i in range(len(qpos))}
        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        if self.real_robot.name == "so100_follower":
            qpos["elbow_flex.pos"] = qpos["elbow_flex.pos"] + 6.8
        elif self.real_robot.name == "so101_follower":
            # Convert gripper from sim degrees to servo degrees
            sim_deg = qpos["gripper.pos"]
            qpos["gripper.pos"] = (sim_deg - self._gripper_sim_min) / self._gripper_sim_range * self._gripper_servo_range + self._gripper_servo_min
        # Drop writes to disabled/overloaded motors so we don't push them into protection mode
        for bad in self._disabled_motors | self._overload_motors:
            key = f"{bad}.pos"
            if key in qpos:
                del qpos[key]
        if not qpos:
            return
        bus = self.real_robot.bus
        try:
            if hasattr(bus, "port_handler") and bus.port_handler.is_open:
                bus.port_handler.clearPort()
        except Exception:
            pass
        self.real_robot.send_action(qpos)

    def reset(self, qpos: Array):
        qpos = common.to_cpu_tensor(qpos)
        freq = 30
        target_pos = self.qpos
        max_rad_per_step = 0.025
        for _ in range(int(20 * freq)):
            start_loop_t = time.perf_counter()
            delta_step = (qpos - target_pos).clip(
                min=-max_rad_per_step, max=max_rad_per_step
            )
            if np.linalg.norm(delta_step) <= 1e-4:
                break
            target_pos += delta_step

            self.set_target_qpos(target_pos)
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(1 / freq - dt_s)
        # Let motors settle after movement so subsequent reads don't hit bus contention
        time.sleep(0.15)
        try:
            bus = self.real_robot.bus
            if hasattr(bus, "port_handler") and bus.port_handler.is_open:
                bus.port_handler.clearPort()
        except Exception:
            pass

    def capture_sensor_data(self, sensor_names: Optional[list[str]] = None):
        sensor_obs = dict()
        cameras: dict[str, Camera] = self.real_robot.cameras
        if sensor_names is None:
            sensor_names = list(cameras.keys())
        for name in sensor_names:
            camera = cameras[name]
            data = camera.async_read()
            sensor_obs[name] = dict(rgb=(common.to_tensor(data)).unsqueeze(0))
        self._captured_sensor_data = sensor_obs

    def get_sensor_data(self, sensor_names: Optional[list[str]] = None):
        if self._captured_sensor_data is None:
            raise RuntimeError(
                "No sensor data captured yet. Please call capture_sensor_data() first."
            )
        if sensor_names is None:
            return self._captured_sensor_data
        else:
            return {
                k: v for k, v in self._captured_sensor_data.items() if k in sensor_names
            }

    def get_qpos(self):
        # NOTE (stao): the slowest part of inference is reading the qpos from the robot. Each time it takes about 5-6 milliseconds, meaning control frequency is capped at 200Hz.
        # and if you factor in other operations like policy inference etc. the max control frequency is typically more like 30-60 Hz.
        # Moreover on the rare occassions reading qpos can take 40 milliseconds which causes the control step to fall behind the desired control frequency.
        if self.use_cached_qpos and self._cached_qpos is not None:
            return self._cached_qpos.clone()
        qpos_deg = {}
        bus = self.real_robot.bus
        # Iterate the fixed 6-motor order so wrist_roll (removed from bus.motors) is still represented
        for name in self._all_motor_keys:
            # Disabled motors: return 0.0 without touching the bus
            if name in self._disabled_motors:
                qpos_deg[name] = 0.0
                continue
            # Motors removed from bus.motors (e.g. physically dead): also return 0.0
            if name not in bus.motors:
                qpos_deg[name] = 0.0
                continue
            for attempt in range(20):
                try:
                    if hasattr(bus, "port_handler") and bus.port_handler.is_open:
                        try:
                            bus.port_handler.clearPort()
                        except Exception:
                            pass
                    qpos_deg[name] = bus.read("Present_Position", name, num_retry=8)
                    break
                except RuntimeError as e:
                    # Overload error -> add to overload set so future writes are skipped
                    if "Overload" in str(e):
                        print(f"[WARN] Motor '{name}' overload detected - disabling writes to this motor")
                        self._overload_motors.add(name)
                        qpos_deg[name] = 0.0
                        break
                    if attempt == 19:
                        raise
                    time.sleep(0.05 + 0.1 * attempt)
                except Exception as e:
                    if attempt == 19:
                        raise
                    time.sleep(0.05 + 0.1 * attempt)

        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        if self.real_robot.name == "so100_follower":
            qpos_deg["elbow_flex"] = qpos_deg["elbow_flex"] - 6.8
        elif self.real_robot.name == "so101_follower":
            # Convert gripper from servo range to sim degrees
            servo_val = qpos_deg["gripper"]
            qpos_deg["gripper"] = (servo_val - self._gripper_servo_min) / self._gripper_servo_range * self._gripper_sim_range + self._gripper_sim_min
        if self._motor_keys is None:
            self._motor_keys = list(qpos_deg.keys())
        qpos_deg = common.flatten_state_dict(qpos_deg)
        qpos = torch.deg2rad(torch.tensor(qpos_deg)).unsqueeze(0)
        self._cached_qpos = qpos
        return qpos

    def get_qvel(self):
        raise NotImplementedError
