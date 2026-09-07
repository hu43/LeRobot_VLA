"""Record a video of the fold_cloth environment rollout.

Runs in headless mode with cameras enabled. Loads best_ckpt.pt if available,
otherwise uses a random policy. Captures the `top` camera frames and saves
as mp4 via imageio.

Usage:
    python scripts/training/record_video.py \
        --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
        --num_envs 1 --num_frames 300 --output runs/video.mp4
"""

from __future__ import annotations

# NOTE: must run before any omniverse/numpy imports. IsaacSim internals call
# np._no_nep50_warning() which was removed in numpy 2.x. Provide a stub.
import numpy as _np
from contextlib import contextmanager
if not hasattr(_np, "_no_nep50_warning"):
    @contextmanager
    def _no_nep50_warning():
        yield
    _np._no_nep50_warning = _no_nep50_warning
    _np._set_string_type = lambda *a, **k: None  # safety net for other nep50 attrs

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record fold_cloth rollout video.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-FoldCloth-BiArm-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=300)
parser.add_argument("--ckpt", type=str, default=None, help="path to best_ckpt.pt")
parser.add_argument("--output", type=str, default="runs/video.mp4")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# enable cameras - we need them for video capture
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import leisaac  # noqa: F401 - register gym tasks
import numpy as np
import torch
import torch.nn as nn
from isaaclab_tasks.utils import parse_env_cfg


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class StatePolicy(nn.Module):
    """MLP policy for joint-state-only observations (must match train_ppo.py)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_action_and_value(self, x, action=None):
        feats = self.feature_net(x)
        action_mean = self.actor_mean(feats)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.normal.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        return action, None, None, None


def flatten_obs(obs: dict, device: torch.device) -> torch.Tensor:
    keys_to_keep = [
        "left_joint_pos", "left_joint_pos_target",
        "right_joint_pos", "right_joint_pos_target",
        "actions",
    ]
    tensors = []
    for k in keys_to_keep:
        if k in obs:
            tensors.append(obs[k].float())
    if not tensors:
        for k, v in obs.items():
            if v.ndim >= 2 and v.shape[-1] <= 32:
                tensors.append(v.float())
    return torch.cat(tensors, dim=-1).to(device)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    # keep top camera enabled, drop wrist cameras to save sim time
    for cam_name in ["left_wrist", "right_wrist"]:
        if hasattr(env_cfg.scene, cam_name):
            setattr(env_cfg.scene, cam_name, None)
        if cam_name in env_cfg.observation_space:
            env_cfg.observation_space.pop(cam_name, None)
        if cam_name in env_cfg.state_space:
            env_cfg.state_space.pop(cam_name, None)
    env_cfg.cameras = []
    env_cfg.use_teleop_device("bi_so101_state_machine")
    env_cfg.recorders = None
    env_cfg.seed = args_cli.seed
    env_cfg.never_time_out = False
    env_cfg.auto_terminate = False
    env_cfg.manual_terminate = False
    env_cfg.return_success_status = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # disable gravity on Robot links (matches train_ppo.py setup)
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if hasattr(env, "initialize"):
        env.initialize()

    obs, _ = env.reset()
    flat_obs = flatten_obs(obs["policy"], device)
    obs_dim = flat_obs.shape[-1]
    action_dim = int(np.prod(env.action_space.shape))
    print(f"[record] obs_dim={obs_dim}, action_dim={action_dim}")

    agent = None
    if args_cli.ckpt and os.path.isfile(args_cli.ckpt):
        agent = StatePolicy(obs_dim, action_dim).to(device)
        sd = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and any("feature_net" in k for k in sd):
            agent.load_state_dict(sd)
        else:
            agent.load_state_dict(sd)
        agent.eval()
        print(f"[record] loaded ckpt {args_cli.ckpt}")
    else:
        print(f"[record] no ckpt ({args_cli.ckpt}), using random policy")

    # get top camera from scene
    try:
        top_cam = env.scene["top"]
    except KeyError:
        raise RuntimeError("top camera not found in scene - cannot record video")
    print(f"[record] top cam: {type(top_cam).__name__}")

    frames = []
    next_obs = obs["policy"]
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    start = time.time()
    for step in range(args_cli.num_frames):
        if agent is not None:
            with torch.no_grad():
                action = agent.get_action_and_value(flat_obs)[0]
        else:
            action = torch.zeros((args_cli.num_envs, action_dim), device=device)
            # small periodic perturbation so motion is visible
            t = step * 0.1
            action[:, 0] = 0.3 * float(np.sin(t))
            action[:, 6] = 0.3 * float(np.cos(t))

        obs, reward, term, trunc, info = env.step(action)
        flat_obs = flatten_obs(obs["policy"], device)

        # capture RGBA from top camera
        rgba = top_cam.data.output["rgba"][0].cpu().numpy()  # (H, W, 4)
        rgb = rgba[..., :3].astype(np.uint8)
        frames.append(rgb)

        if (step + 1) % 30 == 0:
            print(f"[record] {step + 1}/{args_cli.num_frames} frames, reward={reward.mean().item():.3f}")

    elapsed = time.time() - start
    print(f"[record] captured {len(frames)} frames in {elapsed:.1f}s ({len(frames) / elapsed:.1f} fps)")

    out_path = Path(args_cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=30)
    print(f"[record] saved video to {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
