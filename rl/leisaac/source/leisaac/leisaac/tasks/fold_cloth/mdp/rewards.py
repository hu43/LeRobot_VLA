"""Reward functions for the fold cloth task.

Dense reward design (stage-wise):
- approach: exp(-sigma * dist(ee, cloth_edge)) encourages arms to move towards cloth edges
- grasp: binary reward when both arms are close to cloth grasp points
- fold: dense reward based on distances between cloth keypoint pairs (the smaller the more folded)
- rest: binary reward when both arms return to rest pose (after folding)
- coverage_penalty: penalize cloth coverage decrease (cloth should stay on table)
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from leisaac.enhance.assets import ClothObject
from leisaac.utils.robot_utils import is_so101_at_rest_pose


def _get_cloth_keypoints(env: ManagerBasedRLEnv | DirectRLEnv, cloth_cfg: SceneEntityCfg, keypoint_indices: list[int]) -> torch.Tensor:
    """Return positions of the selected cloth keypoints, shape (num_envs, N, 3)."""
    cloth: ClothObject = env.scene.particle_objects[cloth_cfg.name]
    points = cloth.point_positions
    num_envs = env.num_envs
    # points may be (total_particles, 3) or (num_envs, num_particles, 3)
    if points.ndim == 2:
        num_particles = points.shape[0] // num_envs
        points = points.view(num_envs, num_particles, 3)
    elif points.ndim == 3:
        pass  # already (num_envs, num_particles, 3)
    else:
        raise RuntimeError(f"unexpected cloth points shape: {points.shape}")
    idx = torch.tensor(keypoint_indices, device=env.device, dtype=torch.long)
    return points[:, idx, :]


def approach_cloth_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloths"),
    grasp_keypoint_indices: list[int] | None = None,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Dense approach reward: exp(-sigma * min_dist(arm_ee, cloth_grasp_point)).

    Uses the body positions of the last link of each arm as end-effector position.
    """
    if grasp_keypoint_indices is None:
        # default to left sleeve and right sleeve as grasp points
        grasp_keypoint_indices = [159789, 159716]

    left_arm: Articulation = env.scene[left_arm_cfg.name]
    right_arm: Articulation = env.scene[right_arm_cfg.name]

    # use last body link position as ee position
    left_ee_pos = left_arm.data.body_pos_w[:, -1, :] - env.scene.env_origins
    right_ee_pos = right_arm.data.body_pos_w[:, -1, :] - env.scene.env_origins

    keypoints = _get_cloth_keypoints(env, cloth_cfg, grasp_keypoint_indices)  # (N, K, 3)
    # for each arm, find the closest keypoint
    left_dist = torch.linalg.vector_norm(
        keypoints[:, 0:1, :] - left_ee_pos.unsqueeze(1), dim=2
    ).min(dim=1).values
    right_dist = torch.linalg.vector_norm(
        keypoints[:, 1:2, :] - right_ee_pos.unsqueeze(1), dim=2
    ).min(dim=1).values

    return 0.5 * torch.exp(-sigma * left_dist) + 0.5 * torch.exp(-sigma * right_dist)


def grasp_cloth_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloths"),
    grasp_keypoint_indices: list[int] | None = None,
    dist_threshold: float = 0.10,
) -> torch.Tensor:
    """Binary reward when both arms are close to their target cloth grasp points."""
    if grasp_keypoint_indices is None:
        grasp_keypoint_indices = [159789, 159716]

    left_arm: Articulation = env.scene[left_arm_cfg.name]
    right_arm: Articulation = env.scene[right_arm_cfg.name]

    left_ee_pos = left_arm.data.body_pos_w[:, -1, :] - env.scene.env_origins
    right_ee_pos = right_arm.data.body_pos_w[:, -1, :] - env.scene.env_origins

    keypoints = _get_cloth_keypoints(env, cloth_cfg, grasp_keypoint_indices)
    left_dist = torch.linalg.vector_norm(keypoints[:, 0, :] - left_ee_pos, dim=1)
    right_dist = torch.linalg.vector_norm(keypoints[:, 1, :] - right_ee_pos, dim=1)

    return torch.logical_and(left_dist < dist_threshold, right_dist < dist_threshold).float()


def fold_cloth_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloths"),
    cloth_keypoint_indices: list[int] | None = None,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Dense fold reward based on distances between cloth keypoint pairs.

    Pairs to bring together (matching the termination condition):
        - left sleeve (0) <-> right shoulder (4)
        - right sleeve (3) <-> left shoulder (1)
        - left hem (2) <-> left shoulder (1)
        - right hem (5) <-> right shoulder (4)
    """
    if cloth_keypoint_indices is None:
        cloth_keypoint_indices = [159789, 120788, 115370, 159716, 121443, 112382]

    keypoints = _get_cloth_keypoints(env, cloth_cfg, cloth_keypoint_indices)  # (N, 6, 3)

    pair_dists = [
        torch.linalg.vector_norm(keypoints[:, 0] - keypoints[:, 4], dim=1),  # left sleeve -> right shoulder
        torch.linalg.vector_norm(keypoints[:, 3] - keypoints[:, 1], dim=1),  # right sleeve -> left shoulder
        torch.linalg.vector_norm(keypoints[:, 2] - keypoints[:, 1], dim=1),  # left hem -> left shoulder
        torch.linalg.vector_norm(keypoints[:, 5] - keypoints[:, 4], dim=1),  # right hem -> right shoulder
    ]
    mean_dist = torch.stack(pair_dists, dim=1).mean(dim=1)
    return torch.exp(-sigma * mean_dist)


def rest_pose_reward(
    env: ManagerBasedRLEnv | DirectRLEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Binary reward when both arms are at rest pose."""
    left_arm: Articulation = env.scene[left_arm_cfg.name]
    right_arm: Articulation = env.scene[right_arm_cfg.name]
    left_rest = is_so101_at_rest_pose(left_arm.data.joint_pos, left_arm.data.joint_names)
    right_rest = is_so101_at_rest_pose(right_arm.data.joint_pos, right_arm.data.joint_names)
    return torch.logical_and(left_rest, right_rest).float()


def cloth_coverage_penalty(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloths"),
    fall_height: float = 0.5,
) -> torch.Tensor:
    """Penalty when the cloth center falls below a threshold (e.g. off the table)."""
    cloth: ClothObject = env.scene.particle_objects[cloth_cfg.name]
    points = cloth.point_positions
    num_envs = env.num_envs
    if points.ndim == 2:
        num_particles = points.shape[0] // num_envs
        points = points.view(num_envs, num_particles, 3)
    # center z per env
    center_z = points[:, :, 2].mean(dim=1) - env.scene.env_origins[:, 2]
    return (center_z < -fall_height).float() * (-1.0)
