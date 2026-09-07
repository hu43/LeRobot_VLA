"""Launch fold_cloth env in GUI mode for live visualization.

Uses env_isaaclab (numpy 1.26.4) which is compatible with IsaacSim cameras
and syntheticdata. Single env to keep GPU/memory load minimal.

Usage:
    PYTHONPATH=/home/zhoulin/hu/lerobot/rl/leisaac/source/leisaac \
    /home/zhoulin/miniconda3/envs/env_isaaclab/bin/python \
    scripts/training/visualize.py --num_envs 1
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize fold_cloth env.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-FoldCloth-BiArm-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--ckpt", type=str, default=None, help="optional policy checkpoint")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=1000, help="max steps before auto-stop")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# GUI mode: no headless, enable cameras for wrist/top view
args_cli.headless = False
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
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
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_action(self, x):
        feats = self.feature_net(x)
        mean = self.actor_mean(feats)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return torch.distributions.normal.Normal(mean, std).sample()


def flatten_obs(obs: dict, device: torch.device) -> torch.Tensor:
    keys_to_keep = [
        "left_joint_pos", "left_joint_pos_target",
        "right_joint_pos", "right_joint_pos_target",
        "actions",
    ]
    tensors = [obs[k].float() for k in keys_to_keep if k in obs]
    if not tensors:
        for k, v in obs.items():
            if v.ndim >= 2 and v.shape[-1] <= 32:
                tensors.append(v.float())
    return torch.cat(tensors, dim=-1).to(device)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    # keep top camera, drop wrist cameras to save GPU
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
    print(f"[viz] obs_dim={obs_dim}, action_dim={action_dim}")

    agent = None
    if args_cli.ckpt and os.path.isfile(args_cli.ckpt):
        agent = StatePolicy(obs_dim, action_dim).to(device)
        sd = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
        # drop critic keys - inference only needs actor + feature_net
        sd = {k: v for k, v in sd.items() if not k.startswith("critic")}
        missing, unexpected = agent.load_state_dict(sd, strict=False)
        if missing:
            print(f"[viz] missing keys: {missing}")
        if unexpected:
            print(f"[viz] unexpected keys: {unexpected}")
        agent.eval()
        print(f"[viz] loaded ckpt {args_cli.ckpt}")
    else:
        print(f"[viz] no ckpt, using scripted pose-hold policy")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    print(f"[viz] running {args_cli.steps} steps. Close IsaacSim window to stop early.")
    for step in range(args_cli.steps):
        if agent is not None:
            with torch.no_grad():
                action = agent.get_action(flat_obs)
        else:
            # gentle periodic motion so user can see arms move
            action = torch.zeros((args_cli.num_envs, action_dim), device=device)
            t = step * 0.05
            for j in range(min(6, action_dim)):
                action[:, j] = 0.2 * float(np.sin(t + j * 0.5))
            for j in range(6, action_dim):
                action[:, j] = 0.2 * float(np.cos(t + (j - 6) * 0.5))

        obs, reward, term, trunc, info = env.step(action)
        flat_obs = flatten_obs(obs["policy"], device)

        if (step + 1) % 30 == 0:
            print(f"[viz] step {step + 1}/{args_cli.steps} reward={reward.mean().item():.3f}")

        # auto-reset on termination
        done = (term | trunc).bool()
        if done.any():
            obs, _ = env.reset()
            flat_obs = flatten_obs(obs["policy"], device)

    print("[viz] done. Close IsaacSim window to exit.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
