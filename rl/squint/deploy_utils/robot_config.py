"""Robot configuration for real deployment. Edit the values below for your setup."""

from pathlib import Path

from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# Motors to skip during connect/handshake (motor 5 / wrist_roll is physically broken).
# Removing from bus.motors means lerobot won't ping it during handshake, and send_action won't try to write it.
DISABLED_MOTORS = ("wrist_roll",)


def create_real_robot() -> Robot:
    """Create and configure a real robot with the specified camera.
    Returns:
        Configured Robot instance
    """
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480
        )},
        # cameras={"base_camera": RealSenseCameraConfig(
        #     serial_number_or_name="053645021390",
        #     fps=30,
        #     width=640,
        #     height=480
        # )},
        id="zihao_follower_arm",
        calibration_dir=Path(__file__).parent,
    )

    robot = make_robot_from_config(robot_config)
    # Drop disabled motors before connect() so the handshake motor check skips them.
    # manipulator.py re-inserts them as 0.0 placeholders to keep the sim's 6-dim qpos ordering.
    for name in DISABLED_MOTORS:
        if name in robot.bus.motors:
            del robot.bus.motors[name]
    # `ids` is a cached_property - invalidate the cache so it recomputes without the deleted motor
    robot.bus.__dict__.pop("ids", None)

    # Monkey-patch write_calibration to skip motors that have been removed from bus.motors.
    # Otherwise lerobot tries to write Homing_Offset to the dead motor and crashes with KeyError.
    _orig_write_calibration = robot.bus.write_calibration
    def _safe_write_calibration(calibration_dict, *args, **kwargs):
        filtered = {k: v for k, v in calibration_dict.items() if k in robot.bus.motors}
        return _orig_write_calibration(filtered, *args, **kwargs)
    robot.bus.write_calibration = _safe_write_calibration

    return robot
