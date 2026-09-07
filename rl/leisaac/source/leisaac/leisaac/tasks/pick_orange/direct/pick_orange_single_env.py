"""Single-orange pick task for RL training.

Simplified version of the full pick_orange task:
- Only one orange (Orange001) + Plate are used
- Orange spawn position is randomized (larger range than multi-orange)
- Dense stage-wise reward: approach -> grasp -> lift -> place -> rest
- State-only observation for fast RL convergence (sim2real RGB training comes later)
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .. import mdp
from ..pick_orange_env_cfg import PickOrangeSceneCfg


@configclass
class PickOrangeSingleEnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for single-orange pick task."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    task_description: str = "Pick one orange and put it on the plate, then return to rest pose."

    # RL hyperparameters
    episode_length_s: float = 10.0  # shorter than multi-orange (25s)
    decimation: int = 2  # control at ~30Hz given 60Hz sim

    def __post_init__(self) -> None:
        super().__post_init__()

        # Only register Orange001 + Plate (skip Orange002/003)
        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=["Orange001", "Plate"]
        )

        # Domain randomization: larger spawn range for Orange001 (user requested)
        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "Orange001",
                    pose_range={
                        "x": (-0.10, 0.10),
                        "y": (-0.10, 0.10),
                        "z": (0.0, 0.0),
                        "yaw": (-180 * torch.pi / 180, 180 * torch.pi / 180),
                    },
                ),
                randomize_object_uniform(
                    "Plate",
                    pose_range={
                        "x": (-0.03, 0.03),
                        "y": (-0.03, 0.03),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.025, 0.025),
                        "y": (-0.025, 0.025),
                        "z": (-0.025, 0.025),
                        "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )


class PickOrangeSingleEnv(SingleArmTaskDirectEnv):
    """Direct env for single-orange pick task with dense RL reward."""

    cfg: PickOrangeSingleEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # Add task-relevant object positions to state for state-only RL training
        orange = self.scene["Orange001"]
        plate = self.scene["Plate"]
        env_origins = self.scene.env_origins
        obs["policy"]["orange_pos"] = orange.data.root_pos_w - env_origins
        obs["policy"]["plate_pos"] = plate.data.root_pos_w - env_origins
        # Subtask flag for monitoring
        obs["subtask_terms"] = {
            "grasp": mdp.grasp_orange_reward(
                self,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg("Orange001"),
            ),
            "place": mdp.place_on_plate_reward(
                self,
                object_cfg=SceneEntityCfg("Orange001"),
                plate_cfg=SceneEntityCfg("Plate"),
            ),
        }
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """Dense stage-wise reward.

        Total reward = approach (dense) + grasp (binary) + lift (binary)
                     + place (binary, large) + rest (binary) - fall_penalty
        """
        approach = mdp.approach_orange_reward(
            self,
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("Orange001"),
            sigma=10.0,
        )
        grasp = mdp.grasp_orange_reward(
            self,
            robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("Orange001"),
        )
        lift = mdp.lift_orange_reward(
            self,
            object_cfg=SceneEntityCfg("Orange001"),
            height_threshold=0.10,
        )
        place = mdp.place_on_plate_reward(
            self,
            object_cfg=SceneEntityCfg("Orange001"),
            plate_cfg=SceneEntityCfg("Plate"),
        )
        rest = mdp.rest_pose_reward(
            self,
            robot_cfg=SceneEntityCfg("robot"),
        )
        fall_penalty = mdp.orange_fell_penalty(
            self,
            object_cfg=SceneEntityCfg("Orange001"),
            fall_height=0.005,
        )
        # Stage-gated reward: rest only counts after place succeeds
        reward = (
            0.1 * approach
            + 0.5 * grasp
            + 1.0 * lift
            + 2.0 * place
            + 1.0 * rest * place  # rest reward gated by place
            + 1.0 * fall_penalty
        )
        return reward

    def _check_success(self) -> torch.Tensor:
        """Success: orange on plate AND robot at rest pose."""
        return mdp.task_done(
            env=self,
            oranges_cfg=[SceneEntityCfg("Orange001")],
            plate_cfg=SceneEntityCfg("Plate"),
        )
