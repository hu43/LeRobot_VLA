"""Reward functions for the pick orange task.

Dense reward design (stage-wise):
- approach: exp(-10 * dist) encourages EE to move towards the orange
- grasp: binary reward when orange is grasped (EE close + gripper closed)
- lift: binary reward when orange is lifted above threshold
- place: binary reward when orange is placed on the plate
- rest: binary reward when robot returns to rest pose (after placing)
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from leisaac.utils.robot_utils import is_so101_at_rest_pose


def approach_orange_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    sigma: float = 10.0,
) -> torch.Tensor:
    """Dense approach reward: exp(-sigma * dist(ee, orange))."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    orange: RigidObject = env.scene[object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 1, :]
    orange_pos = orange.data.root_pos_w
    dist = torch.linalg.vector_norm(orange_pos - ee_pos, dim=1)
    return torch.exp(-sigma * dist)


def grasp_orange_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Binary reward when the orange is grasped (EE close + gripper closed)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    orange: RigidObject = env.scene[object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 1, :]
    orange_pos = orange.data.root_pos_w
    pos_diff = torch.linalg.vector_norm(orange_pos - ee_pos, dim=1)
    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)
    return grasped.float()


def lift_orange_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    height_threshold: float = 0.15,
) -> torch.Tensor:
    """Binary reward when the orange is lifted above threshold (relative to env origin)."""
    orange: RigidObject = env.scene[object_cfg.name]
    orange_height = orange.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (orange_height > height_threshold).float()


def place_on_plate_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Binary reward when the orange is placed on the plate (xy within range, z near plate height)."""
    orange: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_z = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    orange_x = orange.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    orange_y = orange.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    orange_z = orange.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    in_x = torch.logical_and(orange_x < plate_x + x_range[1], orange_x > plate_x + x_range[0])
    in_y = torch.logical_and(orange_y < plate_y + y_range[1], orange_y > plate_y + y_range[0])
    in_z = torch.logical_and(orange_z < plate_z + height_range[1], orange_z > plate_z + height_range[0])
    return torch.logical_and(torch.logical_and(in_x, in_y), in_z).float()


def rest_pose_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary reward when the robot is at rest pose."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos
    joint_names = robot.data.joint_names
    return is_so101_at_rest_pose(joint_pos, joint_names).float()


def orange_fell_penalty(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    fall_height: float = 0.02,
) -> torch.Tensor:
    """Penalty when the orange falls below a threshold (e.g. off the table)."""
    orange: RigidObject = env.scene[object_cfg.name]
    orange_height = orange.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (orange_height < fall_height).float() * (-1.0)
