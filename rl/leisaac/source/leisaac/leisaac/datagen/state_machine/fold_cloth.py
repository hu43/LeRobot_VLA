"""State machine for the fold-cloth task.

This state machine controls two SO101 arms to fold a piece of cloth:
1. Left arm grasps left sleeve, right arm grasps right sleeve
2. Both arms lift and fold the cloth towards the center
3. Arms release and return to rest pose
"""

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRIPPER_OFFSET = 0.08

# Rest pose in degrees for SO101
_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": -10.0,
}


class FoldClothStateMachine(StateMachineBase):
    """State machine for the cloth folding task using two arms."""

    # Total steps for the folding sequence
    MAX_STEPS: int = 1200

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._initial_ee_pos_left: torch.Tensor | None = None
        self._initial_ee_pos_right: torch.Tensor | None = None
        self._rest_ee_pos_left: torch.Tensor | None = None
        self._rest_ee_pos_right: torch.Tensor | None = None
        self._rest_joint_pos_left: torch.Tensor | None = None
        self._rest_joint_pos_right: torch.Tensor | None = None
        self._cloth_left_pos: torch.Tensor | None = None
        self._cloth_right_pos: torch.Tensor | None = None
        self._home_start_left: torch.Tensor | None = None
        self._home_start_right: torch.Tensor | None = None

    def setup(self, env) -> None:
        """Calibrate rest pose EE positions for both arms."""
        # Setup left arm
        left_arm = env.scene["left_arm"]
        left_joint_names = list(left_arm.data.joint_names)

        self._rest_joint_pos_left = torch.zeros(env.num_envs, len(left_joint_names), device=env.device)
        for idx, name in enumerate(left_joint_names):
            if name in _REST_POSE_DEG:
                self._rest_joint_pos_left[:, idx] = _REST_POSE_DEG[name] * torch.pi / 180.0

        left_arm.write_joint_state_to_sim(
            position=self._rest_joint_pos_left,
            velocity=torch.zeros_like(self._rest_joint_pos_left),
        )

        # Setup right arm
        right_arm = env.scene["right_arm"]
        right_joint_names = list(right_arm.data.joint_names)

        self._rest_joint_pos_right = torch.zeros(env.num_envs, len(right_joint_names), device=env.device)
        for idx, name in enumerate(right_joint_names):
            if name in _REST_POSE_DEG:
                self._rest_joint_pos_right[:, idx] = _REST_POSE_DEG[name] * torch.pi / 180.0

        right_arm.write_joint_state_to_sim(
            position=self._rest_joint_pos_right,
            velocity=torch.zeros_like(self._rest_joint_pos_right),
        )

        # Step sim to get FK
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        self._rest_ee_pos_left = left_arm.data.body_pos_w[:, -1, :].clone()
        self._rest_ee_pos_right = right_arm.data.body_pos_w[:, -1, :].clone()

    def check_success(self, env) -> bool:
        """Return True if cloth is folded and both arms are at rest pose."""
        # Teleport to rest pose for checking
        if self._rest_joint_pos_left is not None:
            env.scene["left_arm"].write_joint_state_to_sim(
                position=self._rest_joint_pos_left,
                velocity=torch.zeros_like(self._rest_joint_pos_left),
            )
        if self._rest_joint_pos_right is not None:
            env.scene["right_arm"].write_joint_state_to_sim(
                position=self._rest_joint_pos_right,
                velocity=torch.zeros_like(self._rest_joint_pos_right),
            )

        env.scene.update(dt=env.physics_dt)

        # Use cloth_folded from mdp to check success
        from isaaclab.managers import SceneEntityCfg
        from leisaac.tasks.fold_cloth.mdp import cloth_folded

        success_tensor = cloth_folded(
            env,
            cloth_cfg=SceneEntityCfg("cloths"),
            cloth_keypoints_index=[159789, 120788, 115370, 159716, 121443, 112382],
            distance_threshold=0.20,
        )
        return bool(success_tensor.all().item())

    def pre_step(self, env) -> None:
        """Blend joint state towards rest pose during final phase."""
        step = self._step_count
        if step >= 1000:
            if step == 1000:
                self._home_start_left = env.scene["left_arm"].data.joint_pos.clone()
                self._home_start_right = env.scene["right_arm"].data.joint_pos.clone()

            alpha = min((step - 1000) / 199.0, 1.0)
            if self._rest_joint_pos_left is not None and hasattr(self, "_home_start_left"):
                blended_left = self._home_start_left + (self._rest_joint_pos_left - self._home_start_left) * alpha
                env.scene["left_arm"].write_joint_state_to_sim(
                    position=blended_left,
                    velocity=torch.zeros_like(blended_left),
                )
            if self._rest_joint_pos_right is not None and hasattr(self, "_home_start_right"):
                blended_right = self._home_start_right + (self._rest_joint_pos_right - self._home_start_right) * alpha
                env.scene["right_arm"].write_joint_state_to_sim(
                    position=blended_right,
                    velocity=torch.zeros_like(blended_right),
                )

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for both arms (16D total: 8D left + 8D right)."""
        left_arm = env.scene["left_arm"]
        right_arm = env.scene["right_arm"]

        left_arm.write_joint_damping_to_sim(damping=10.0)
        right_arm.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        # Get cloth keypoint positions
        cloth = env.scene.particle_objects["cloths"]
        cloth_points = cloth.point_positions.clone()

        # Indices: left sleeve, left shoulder, right sleeve, right shoulder
        left_sleeve_idx = 159789
        left_shoulder_idx = 120788
        right_sleeve_idx = 159716
        right_shoulder_idx = 121443

        left_sleeve_pos = cloth_points[:, left_sleeve_idx]
        left_shoulder_pos = cloth_points[:, left_shoulder_idx]
        right_sleeve_pos = cloth_points[:, right_sleeve_idx]
        right_shoulder_pos = cloth_points[:, right_shoulder_idx]

        # Store initial cloth positions on first step
        if step == 0:
            self._cloth_left_pos = left_sleeve_pos.clone()
            self._cloth_right_pos = right_sleeve_pos.clone()
            self._initial_ee_pos_left = left_arm.data.body_pos_w[:, -1, :].clone()
            self._initial_ee_pos_right = right_arm.data.body_pos_w[:, -1, :].clone()

        # Get robot base transforms
        left_base_pos = left_arm.data.root_pos_w.clone()
        left_base_quat = left_arm.data.root_quat_w.clone()
        right_base_pos = right_arm.data.root_pos_w.clone()
        right_base_quat = right_arm.data.root_quat_w.clone()

        # Left arm orientation (slightly rotated for better grasp)
        target_quat_left_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(-torch.pi / 4, device=device),
        ).repeat(num_envs, 1)

        # Right arm orientation (mirror)
        target_quat_right_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(torch.pi / 4, device=device),
        ).repeat(num_envs, 1)

        target_quat_left = quat_mul(quat_inv(left_base_quat), target_quat_left_w)
        target_quat_right = quat_mul(quat_inv(right_base_quat), target_quat_right_w)

        # Execute phases
        if step < 150:
            left_pos, left_grip, right_pos, right_grip = self._phase_approach_cloth(
                left_sleeve_pos, right_sleeve_pos, num_envs, device
            )
        elif step < 250:
            left_pos, left_grip, right_pos, right_grip = self._phase_hover_above_sleeves(
                left_sleeve_pos, right_sleeve_pos, num_envs, device
            )
        elif step < 350:
            left_pos, left_grip, right_pos, right_grip = self._phase_lower_to_sleeves(
                left_sleeve_pos, right_sleeve_pos, num_envs, device
            )
        elif step < 450:
            left_pos, left_grip, right_pos, right_grip = self._phase_grasp_sleeves(
                left_sleeve_pos, right_sleeve_pos, num_envs, device
            )
        elif step < 650:
            left_pos, left_grip, right_pos, right_grip = self._phase_lift_and_fold(
                left_shoulder_pos, right_shoulder_pos, num_envs, device
            )
        elif step < 800:
            left_pos, left_grip, right_pos, right_grip = self._phase_hold_folded(
                left_shoulder_pos, right_shoulder_pos, num_envs, device
            )
        elif step < 900:
            left_pos, left_grip, right_pos, right_grip = self._phase_release(
                left_shoulder_pos, right_shoulder_pos, num_envs, device
            )
        elif step < 1000:
            left_pos, left_grip, right_pos, right_grip = self._phase_lift_away(
                left_shoulder_pos, right_shoulder_pos, num_envs, device
            )
        else:
            left_pos, left_grip, right_pos, right_grip = self._phase_return_home(num_envs, device)

        # Convert to local coordinates for both arms (pos + quat + grip = 8D each)
        diff_left = left_pos - left_base_pos
        left_pos_local = quat_apply(quat_inv(left_base_quat), diff_left)
        diff_right = right_pos - right_base_pos
        right_pos_local = quat_apply(quat_inv(right_base_quat), diff_right)

        # Concatenate left and right arm actions (8D each = 3 pos + 4 quat + 1 gripper)
        left_action = torch.cat([left_pos_local, target_quat_left, left_grip], dim=-1)
        right_action = torch.cat([right_pos_local, target_quat_right, right_grip], dim=-1)

        return torch.cat([left_action, right_action], dim=-1)

    def advance(self) -> None:
        """Advance step counter and mark episode done when finished."""
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        """Reset the state machine for a new episode."""
        self._step_count = 0
        self._episode_done = False
        self._initial_ee_pos_left = None
        self._initial_ee_pos_right = None
        self._cloth_left_pos = None
        self._cloth_right_pos = None
        self._home_start_left = None
        self._home_start_right = None

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _phase_approach_cloth(self, left_sleeve_pos, right_sleeve_pos, num_envs, device):
        """Phase 1: Move both arms from rest pose to hover above cloth sleeves."""
        alpha = self._step_count / 150.0

        # Left target: above left sleeve
        left_target = left_sleeve_pos.clone()
        left_target[:, 2] += 0.2 + _GRIPPER_OFFSET

        # Right target: above right sleeve
        right_target = right_sleeve_pos.clone()
        right_target[:, 2] += 0.2 + _GRIPPER_OFFSET

        # Smooth interpolation from initial EE positions
        if self._initial_ee_pos_left is not None:
            left_pos = (1.0 - alpha) * self._initial_ee_pos_left + alpha * left_target
        else:
            left_pos = left_target

        if self._initial_ee_pos_right is not None:
            right_pos = (1.0 - alpha) * self._initial_ee_pos_right + alpha * right_target
        else:
            right_pos = right_target

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
        )

    def _phase_hover_above_sleeves(self, left_sleeve_pos, right_sleeve_pos, num_envs, device):
        """Phase 2: Hover above the sleeves, fine-tuning position."""
        left_pos = left_sleeve_pos.clone()
        left_pos[:, 2] += 0.15 + _GRIPPER_OFFSET

        right_pos = right_sleeve_pos.clone()
        right_pos[:, 2] += 0.15 + _GRIPPER_OFFSET

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
        )

    def _phase_lower_to_sleeves(self, left_sleeve_pos, right_sleeve_pos, num_envs, device):
        """Phase 3: Lower grippers down to the sleeves."""
        left_pos = left_sleeve_pos.clone()
        left_pos[:, 2] += _GRIPPER_OFFSET

        right_pos = right_sleeve_pos.clone()
        right_pos[:, 2] += _GRIPPER_OFFSET

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
        )

    def _phase_grasp_sleeves(self, left_sleeve_pos, right_sleeve_pos, num_envs, device):
        """Phase 4: Close both grippers to grasp the sleeves."""
        left_pos = left_sleeve_pos.clone()
        left_pos[:, 2] += _GRIPPER_OFFSET

        right_pos = right_sleeve_pos.clone()
        right_pos[:, 2] += _GRIPPER_OFFSET

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device),
        )

    def _phase_lift_and_fold(self, left_shoulder_pos, right_shoulder_pos, num_envs, device):
        """Phase 5: Lift and fold - move both arms towards opposite shoulders."""
        alpha = (self._step_count - 450) / 200.0
        # Use smooth sine interpolation for natural-looking motion
        t = 0.5 * (1.0 - torch.cos(torch.tensor(alpha * torch.pi, device=device)))

        # Left arm moves from left sleeve to right shoulder
        if self._cloth_left_pos is not None:
            left_start = self._cloth_left_pos.clone()
        else:
            left_start = left_shoulder_pos.clone()
            left_start[:, 0] -= 0.3

        left_end = right_shoulder_pos.clone()
        left_end[:, 2] += 0.15

        # Arc height for natural lifting
        lift_height = 0.25
        left_pos = (1.0 - t) * left_start + t * left_end
        left_pos[:, 2] += lift_height * torch.sin(t * torch.tensor(torch.pi, device=device))
        left_pos[:, 2] += _GRIPPER_OFFSET

        # Right arm moves from right sleeve to left shoulder
        if self._cloth_right_pos is not None:
            right_start = self._cloth_right_pos.clone()
        else:
            right_start = right_shoulder_pos.clone()
            right_start[:, 0] += 0.3

        right_end = left_shoulder_pos.clone()
        right_end[:, 2] += 0.15

        right_pos = (1.0 - t) * right_start + t * right_end
        right_pos[:, 2] += lift_height * torch.sin(t * torch.tensor(torch.pi, device=device))
        right_pos[:, 2] += _GRIPPER_OFFSET

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device),
        )

    def _phase_hold_folded(self, left_shoulder_pos, right_shoulder_pos, num_envs, device):
        """Phase 6: Hold the cloth in folded position to let it settle."""
        left_pos = right_shoulder_pos.clone()
        left_pos[:, 2] += 0.15 + _GRIPPER_OFFSET

        right_pos = left_shoulder_pos.clone()
        right_pos[:, 2] += 0.15 + _GRIPPER_OFFSET

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device),
        )

    def _phase_release(self, left_shoulder_pos, right_shoulder_pos, num_envs, device):
        """Phase 7: Release the grippers."""
        left_pos = right_shoulder_pos.clone()
        left_pos[:, 2] += 0.15 + _GRIPPER_OFFSET

        right_pos = left_shoulder_pos.clone()
        right_pos[:, 2] += 0.15 + _GRIPPER_OFFSET

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
        )

    def _phase_lift_away(self, left_shoulder_pos, right_shoulder_pos, num_envs, device):
        """Phase 8: Lift grippers away from the cloth."""
        alpha = (self._step_count - 900) / 100.0

        left_start = right_shoulder_pos.clone()
        left_start[:, 2] += 0.15 + _GRIPPER_OFFSET
        left_end = left_start.clone()
        left_end[:, 2] += 0.15

        right_start = left_shoulder_pos.clone()
        right_start[:, 2] += 0.15 + _GRIPPER_OFFSET
        right_end = right_start.clone()
        right_end[:, 2] += 0.15

        left_pos = (1.0 - alpha) * left_start + alpha * left_end
        right_pos = (1.0 - alpha) * right_start + alpha * right_end

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
        )

    def _phase_return_home(self, num_envs, device):
        """Phase 9: Return both arms to their rest poses."""
        if self._rest_ee_pos_left is not None:
            left_pos = self._rest_ee_pos_left.clone()
        elif self._initial_ee_pos_left is not None:
            left_pos = self._initial_ee_pos_left.clone()
        else:
            left_pos = torch.zeros(num_envs, 3, device=device)

        if self._rest_ee_pos_right is not None:
            right_pos = self._rest_ee_pos_right.clone()
        elif self._initial_ee_pos_right is not None:
            right_pos = self._initial_ee_pos_right.clone()
        else:
            right_pos = torch.zeros(num_envs, 3, device=device)

        return (
            left_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
            right_pos,
            torch.full((num_envs, 1), _GRIPPER_OPEN, device=device),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
