# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XLeRobot VR teleoperator (browser WebXR, e.g. Meta Quest).

Ported from XLeRobot/software/src/teleporators/xlerobot_vr/xlerobot_vr.py to the
current Teleoperator API. Robot observations are pushed into this teleoperator
via `update_observation(obs)` (called by lerobot-teleoperate / lerobot-record)
and are used for the joint-space P-control and controller initialization.
"""

import asyncio
import logging
import threading
import time
from typing import Any

from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_xlerobot_vr import XLerobotVRTeleopConfig
from .vr_monitor import VRMonitor, get_local_ip

logger = logging.getLogger(__name__)

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}

RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class SimpleTeleopArm:
    """Delta-action VR control of one arm with 2-link IK and P-control."""

    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1.0):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics

        self.joint_positions = {
            joint: initial_obs.get(f"{prefix}_arm_{joint}.pos", 0.0) for joint in ARM_JOINTS
        }

        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        self.last_vr_time = 0.0
        self.max_delta_per_frame = 0.005

        self.target_positions = {joint: 0.0 for joint in ARM_JOINTS}
        self.zero_pos = {joint: 0.0 for joint in ARM_JOINTS}

    def move_to_zero_position(self, robot_obs):
        logger.info(f"[{self.prefix}] Moving to zero position")
        self.target_positions = self.zero_pos.copy()
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        self.last_vr_time = 0.0
        self.target_positions["wrist_flex"] = 0.0
        return self.p_control_action(robot_obs)

    def handle_vr_input(self, vr_goal, gripper_state=None):
        if vr_goal is None:
            return
        if not hasattr(vr_goal, "target_position") or vr_goal.target_position is None:
            return

        current_vr_pos = vr_goal.target_position

        if not hasattr(self, "prev_vr_pos"):
            self.prev_vr_pos = current_vr_pos
            return

        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * 170
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * 80
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * 80

        self.prev_vr_pos = current_vr_pos

        pos_scale = 0.015
        angle_scale = 3.0
        delta_limit = 0.02
        angle_limit = 6.0

        delta_x = vr_x * pos_scale
        delta_y = vr_y * pos_scale
        delta_z = vr_z * pos_scale

        threshold = 0.001
        if abs(delta_x) < threshold:
            delta_x = 0.0
        if abs(delta_y) < threshold:
            delta_y = 0.0
        if abs(delta_z) < threshold:
            delta_z = 0.0

        delta_x = max(-delta_limit, min(delta_limit, delta_x))
        delta_y = max(-delta_limit, min(delta_limit, delta_y))
        delta_z = max(-delta_limit, min(delta_limit, delta_z))

        self.current_x += -delta_z
        self.current_y += delta_y

        if hasattr(vr_goal, "wrist_flex_deg") and vr_goal.wrist_flex_deg is not None:
            if not hasattr(self, "prev_wrist_flex"):
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
                return

            delta_pitch = (vr_goal.wrist_flex_deg - self.prev_wrist_flex) * angle_scale
            if abs(delta_pitch) < 1:
                delta_pitch = 0.0
            delta_pitch = max(-angle_limit, min(angle_limit, delta_pitch))
            self.pitch += delta_pitch
            self.pitch = max(-90, min(90, self.pitch))
            self.prev_wrist_flex = vr_goal.wrist_flex_deg

        if hasattr(vr_goal, "wrist_roll_deg") and vr_goal.wrist_roll_deg is not None:
            if not hasattr(self, "prev_wrist_roll"):
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
                return

            delta_roll = (vr_goal.wrist_roll_deg - self.prev_wrist_roll) * angle_scale
            delta_roll = max(-angle_limit, min(angle_limit, delta_roll))
            if abs(delta_roll) < 1:
                delta_roll = 0.0

            current_roll = self.target_positions.get("wrist_roll", 0.0)
            new_roll = max(-90, min(90, current_roll + delta_roll))
            self.target_positions["wrist_roll"] = new_roll
            self.prev_wrist_roll = vr_goal.wrist_roll_deg

        if abs(delta_x) > 0.001:
            delta_pan = max(-angle_limit, min(angle_limit, delta_x * 200.0))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            self.target_positions["shoulder_pan"] = max(-180, min(180, current_pan + delta_pan))

        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            alpha = 0.1
            self.target_positions["shoulder_lift"] = (
                (1 - alpha) * self.target_positions.get("shoulder_lift", 0.0) + alpha * joint2_target
            )
            self.target_positions["elbow_flex"] = (
                (1 - alpha) * self.target_positions.get("elbow_flex", 0.0) + alpha * joint3_target
            )
        except Exception as e:
            logger.debug(f"[{self.prefix}] VR IK failed: {e}")

        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch
        )

        if vr_goal.metadata and vr_goal.metadata.get("trigger", 0) > 0.5:
            self.target_positions["gripper"] = 45
        else:
            self.target_positions["gripper"] = 0.0

    def p_control_action(self, robot_obs) -> dict[str, float]:
        current = {joint: robot_obs.get(f"{self.prefix}_arm_{joint}.pos", 0.0) for joint in ARM_JOINTS}
        action = {}
        for joint in self.target_positions:
            error = self.target_positions[joint] - current[joint]
            control = self.kp * error
            action[f"{self.joint_map[joint]}.pos"] = current[joint] + control
        return action


class SimpleHeadControl:
    """Holds the head motors at their initial position (no active VR mapping)."""

    def __init__(self, initial_obs, kp=1.0):
        self.kp = kp
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot_obs):
        self.target_positions = self.zero_pos.copy()
        return self.p_control_action(robot_obs)

    def p_control_action(self, robot_obs) -> dict[str, float]:
        action = {}
        for motor in self.target_positions:
            current = robot_obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + self.kp * error
        return action


class VREventHandler:
    """Left-controller thumbstick events for dataset recording control.

    - thumbstick left: re-record current episode
    - thumbstick right: exit current loop early
    - thumbstick up: stop recording
    - thumbstick down: reset robot position
    """

    def __init__(self):
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "reset_position": False,
            "back_position": False,
        }
        self.prev_states = {"thumbstick_x": 0, "thumbstick_y": 0, "trigger": False}
        self.threshold = 0.7

    def process_left_controller(self, metadata):
        if not metadata:
            return

        thumb = metadata.get("thumbstick", {})
        thumb_x = thumb.get("x", 0)
        thumb_y = thumb.get("y", 0)

        if thumb_x > self.threshold and self.prev_states["thumbstick_x"] <= self.threshold:
            logger.info("VR left controller right -> exit loop early")
            self.events["exit_early"] = True
        elif thumb_x < -self.threshold:
            logger.info("VR left controller left -> re-record episode")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True

        if thumb_y > self.threshold and self.prev_states["thumbstick_y"] <= self.threshold:
            logger.info("VR left controller up -> stop recording")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
        elif thumb_y < -self.threshold and self.prev_states["thumbstick_y"] >= -self.threshold:
            logger.info("VR left controller down -> reset robot position")
            self.events["reset_position"] = True
        else:
            self.events["reset_position"] = False
            self.events["back_position"] = False

        trigger = metadata.get("trigger", 0) > 0.5

        self.prev_states.update(
            {"thumbstick_x": thumb_x, "thumbstick_y": thumb_y, "trigger": trigger}
        )

    def get_events(self):
        return self.events.copy()

    def reset_events(self):
        for key in self.events:
            self.events[key] = False


class XLerobotVRTeleop(Teleoperator):
    """XLeRobot VR teleoperator producing the 17-key xlerobot action space."""

    config_class = XLerobotVRTeleopConfig
    name = "xlerobot_vr"

    def __init__(self, config: XLerobotVRTeleopConfig):
        super().__init__(config)
        self.config = config

        self.vr_monitor: VRMonitor | None = None
        self.vr_thread: threading.Thread | None = None

        self.kin_left = SO101Kinematics()
        self.kin_right = SO101Kinematics()

        self.left_arm: SimpleTeleopArm | None = None
        self.right_arm: SimpleTeleopArm | None = None
        self.head_control: SimpleHeadControl | None = None
        self.vr_event_handler: VREventHandler | None = None

        self._last_obs: dict[str, Any] | None = None
        self._last_action: dict[str, float] = {}
        self._last_event_update_time = 0.0

        self.speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self.speed_index = 0

        self._connected = False
        self.logs = {}

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        for joint_name in LEFT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = float
        for joint_name in RIGHT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = float
        for motor_name in HEAD_MOTOR_MAP.values():
            features[f"{motor_name}.pos"] = float
        features["x.vel"] = float
        features["y.vel"] = float
        features["theta.vel"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self.vr_monitor is not None
            and self.vr_thread is not None
            and self.vr_thread.is_alive()
        )

    @property
    def is_calibrated(self) -> bool:
        # VR controllers need no calibration; arm controllers are initialized
        # from the first robot observation instead.
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.vr_monitor = VRMonitor(self.config.xlevr_path, display_host=self.config.display_host)

        if not self.vr_monitor.initialize():
            raise RuntimeError(
                f"Failed to initialize VR monitor from XLeVR path: {self.config.xlevr_path}. "
                "Check the path and install XLeVR dependencies (uv pip install websockets scipy)."
            )

        self.vr_thread = threading.Thread(
            target=lambda: asyncio.run(self.vr_monitor.start_monitoring()),
            daemon=True,
        )
        self.vr_thread.start()
        time.sleep(0.5)

        if not self.vr_thread.is_alive():
            raise RuntimeError("VR monitoring thread failed to start")

        self._connected = True
        self.vr_event_handler = VREventHandler()

        host = self.vr_monitor.config.host_ip
        https_port = self.vr_monitor.config.https_port

        display_host = self.config.display_host or (get_local_ip() if host == "0.0.0.0" else host)
        logger.info(
            f"Waiting for VR connection. Open your VR headset browser at: "
            f"https://{display_host}:{https_port} "
            f"(timeout: {self.config.vr_connection_timeout}s)"
        )

        start_time = time.time()
        while not self.vr_monitor.has_vr_data():
            if time.time() - start_time > self.config.vr_connection_timeout:
                logger.warning(
                    "Timed out waiting for VR data. Continuing anyway - "
                    "the robot will stay still until the VR controllers connect."
                )
                break
            time.sleep(0.5)

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def update_observation(self, obs: dict[str, Any]) -> None:
        """Receive the latest robot observation (called by teleoperate/record loops)."""
        self._last_obs = obs
        if self.left_arm is None:
            self._init_controllers()

    def _init_controllers(self) -> None:
        obs = self._last_obs if self._last_obs is not None else {}
        self.left_arm = SimpleTeleopArm(
            LEFT_JOINT_MAP, obs, self.kin_left, prefix="left", kp=self.config.kp
        )
        self.right_arm = SimpleTeleopArm(
            RIGHT_JOINT_MAP, obs, self.kin_right, prefix="right", kp=self.config.kp
        )
        self.head_control = SimpleHeadControl(obs, kp=self.config.kp)
        logger.info("VR arm/head controllers initialized")

    def _get_base_action(self, right_goal) -> dict[str, float]:
        # Mirrors XLerobot._from_keyboard_to_base_action with the VR mapping
        # from XLeRobot: right thumbstick y = forward/backward,
        # right thumbstick x = rotate left/right.
        speed = self.speed_levels[self.speed_index]
        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if right_goal is not None and getattr(right_goal, "metadata", None):
            thumb = right_goal.metadata.get("thumbstick", {})
            thumb_x = thumb.get("x", 0)
            thumb_y = thumb.get("y", 0)
            if abs(thumb_x) > 0.2:
                # 'u' = rotate_left (theta +), 'o' = rotate_right (theta -)
                theta_cmd = -speed["theta"] if thumb_x > 0 else speed["theta"]
            if abs(thumb_y) > 0.2:
                # 'i' = forward (x +), 'k' = backward (x -)
                x_cmd = -speed["xy"] if thumb_y > 0 else speed["xy"]

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    def update_events(self, events: dict) -> None:
        """Push VR recording events into the shared events dict (record loop)."""
        if self.vr_event_handler is None:
            return

        dual_goals = self.vr_monitor.get_latest_goal_nowait() if self.vr_monitor else None
        left_goal = dual_goals.get("left") if dual_goals else None
        if left_goal is not None and getattr(left_goal, "metadata", None):
            self.vr_event_handler.process_left_controller(left_goal.metadata)

        vr_events = self.vr_event_handler.get_events()
        for key in ("exit_early", "rerecord_episode", "stop_recording"):
            if vr_events.get(key):
                events[key] = True

        # One-shot events are consumed once reported.
        if vr_events.get("exit_early") or vr_events.get("rerecord_episode"):
            self.vr_event_handler.reset_events()

    def get_teleop_events(self) -> dict:
        """Snapshot of the VR events (mirrors KeyboardTeleop-style accessors)."""
        if self.vr_event_handler is None:
            return VREventHandler().get_events()
        return self.vr_event_handler.get_events()

    def move_to_zero_position(self) -> dict[str, float]:
        obs = self._last_obs or {}
        action = {}
        if self.left_arm:
            action.update(self.left_arm.move_to_zero_position(obs))
        if self.right_arm:
            action.update(self.right_arm.move_to_zero_position(obs))
        if self.head_control:
            action.update(self.head_control.move_to_zero_position(obs))
        action.update(self._get_base_action(None))
        return action

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        before_read_t = time.perf_counter()

        if self.left_arm is None:
            self._init_controllers()

        if self._last_obs is None:
            # Without an observation we cannot P-control; wait for
            # update_observation() from the teleoperate/record loop.
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return self._last_action

        if self.vr_monitor is None:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return self._last_action

        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
        except Exception as e:
            logger.warning(f"VR data acquisition failed: {e}")
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return self._last_action

        left_goal = dual_goals.get("left")
        right_goal = dual_goals.get("right")

        obs = self._last_obs if self._last_obs is not None else {}
        action: dict[str, float] = {}

        try:
            if left_goal is not None:
                self.left_arm.handle_vr_input(left_goal, None)
            if right_goal is not None:
                self.right_arm.handle_vr_input(right_goal, None)

            action.update(self.left_arm.p_control_action(obs))
            action.update(self.right_arm.p_control_action(obs))
            action.update(self.head_control.p_control_action(obs))
            action.update(self._get_base_action(right_goal))
        except Exception as e:
            logger.error(f"VR action generation failed: {e}")
            return self._last_action

        self._last_action = action
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.vr_monitor:
            self.vr_monitor.stop()
        self._connected = False
        logger.info(f"{self} disconnected.")
