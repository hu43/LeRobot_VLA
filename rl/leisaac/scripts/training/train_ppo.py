"""Train a PPO agent on leisaac fold cloth (or any leisaac DirectRLEnv) task.

Usage:
    python scripts/training/train_ppo.py --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
        --num_envs 64 --total_timesteps 5000000

This is a self-contained PPO implementation targeting leisaac DirectRLEnv,
independent of ManiSkill. It collects rollouts from the vec env, trains a
joint state-only (no image) policy, and saves checkpoints under runs/<run_name>/.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# NOTE: numpy is imported after AppLauncher boots. In headless mode, Isaac Sim's
# bundled numpy 1.x works fine with gymnasium. In GUI mode there is a conflict
# we haven't resolved yet, so we recommend --headless.

# launch isaac sim first
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO training on leisaac DirectRLEnv.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-FoldCloth-BiArm-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--total_timesteps", type=int, default=5_000_000)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--num_steps", type=int, default=64, help="rollout steps per env per iteration")
parser.add_argument("--num_minibatches", type=int, default=16)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.85)
parser.add_argument("--gae_lambda", type=float, default=0.9)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--target_kl", type=float, default=0.2)
parser.add_argument("--norm_adv", action="store_true", default=True)
parser.add_argument("--anneal_lr", action="store_true", default=False)
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=20, help="save checkpoint every N iterations")
parser.add_argument("--eval_freq", type=int, default=10, help="evaluate every N iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# disable cameras for state-based PPO (numpy 2.x conflicts with syntheticdata)
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = False

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac  # noqa: F401 - register gym tasks
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from isaaclab_tasks.utils import parse_env_cfg
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class PPOConfig:
    task: str = "LeIsaac-SO101-FoldCloth-BiArm-Direct-v0"
    num_envs: int = 64
    seed: int = 1
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_steps: int = 64
    num_minibatches: int = 16
    update_epochs: int = 4
    gamma: float = 0.85
    gae_lambda: float = 0.9
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    norm_adv: bool = True
    anneal_lr: bool = False
    exp_name: Optional[str] = None
    save_freq: int = 20
    eval_freq: int = 10
    device: str = "cuda:0"
    headless: bool = False


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class StatePolicy(nn.Module):
    """MLP policy for joint-state-only observations (concatenated flat dict)."""

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

    def get_value(self, x):
        return self.critic(self.feature_net(x))

    def get_action_and_value(self, x, action=None):
        feats = self.feature_net(x)
        action_mean = self.actor_mean(feats)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(feats)
        return action, log_prob, entropy, value


def flatten_obs(obs: dict, device: torch.device) -> torch.Tensor:
    """Concatenate the joint-state observation keys into a flat tensor.

    Drops image keys (left_wrist/right_wrist/top) since this is a state-based PPO.
    """
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
        # fallback: concatenate any non-image fields
        for k, v in obs.items():
            if v.ndim >= 2 and v.shape[-1] <= 32:
                tensors.append(v.float())
    return torch.cat(tensors, dim=-1).to(device)


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            step_idx, key = idx
            getattr(self, key)[step_idx] = value
        else:
            raise IndexError(idx)


def evaluate(env, agent: StatePolicy, device: torch.device, num_eval_episodes: int = 5):
    """Run several deterministic rollouts and return mean reward and success rate."""
    obs, _ = env.reset()
    total_reward = 0.0
    successes = 0
    episodes = 0
    steps_per_episode = 0
    while episodes < num_eval_episodes:
        flat = flatten_obs(obs["policy"], device)
        with torch.no_grad():
            action = agent.get_action_and_value(flat)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.mean().item()
        steps_per_episode += 1
        done_mask = (terminated | truncated).bool()
        if done_mask.any():
            for k in range(done_mask.shape[0]):
                if done_mask[k]:
                    episodes += 1
                    if "success" in info:
                        successes += int(info["success"][k].item())
                    break
            obs, _ = env.reset()
            steps_per_episode = 0
    return total_reward / max(episodes, 1), successes / max(episodes, 1)


def train(cfg: PPOConfig):
    exp_name = cfg.exp_name or f"{cfg.task.split('/')[-1]}__ppo__{cfg.seed}__{int(time.time())}"
    run_dir = Path("runs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    device_str = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"[ppo] device: {device} (str: {device_str})")

    # build env
    env_cfg = parse_env_cfg(cfg.task, device=device_str, num_envs=cfg.num_envs)
    # state-based PPO: drop cameras (they conflict with numpy 2.x in syntheticdata)
    for cam_name in ["left_wrist", "right_wrist", "top"]:
        if hasattr(env_cfg.scene, cam_name):
            setattr(env_cfg.scene, cam_name, None)
        if cam_name in env_cfg.observation_space:
            env_cfg.observation_space.pop(cam_name, None)
        if cam_name in env_cfg.state_space:
            env_cfg.state_space.pop(cam_name, None)
    env_cfg.cameras = []
    # state-machine style setup: configure teleop device so gravity is disabled on arms
    env_cfg.use_teleop_device("bi_so101_state_machine")
    env_cfg.recorders = None
    env_cfg.seed = cfg.seed
    env_cfg.never_time_out = False
    env_cfg.auto_terminate = False
    env_cfg.manual_terminate = False
    env_cfg.return_success_status = False
    env = gym.make(cfg.task, cfg=env_cfg).unwrapped

    # disable gravity on all Robot link prims (matches generate.py setup)
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if hasattr(env, "initialize"):
        env.initialize()

    # probe dims
    obs, _ = env.reset()
    flat_obs = flatten_obs(obs["policy"], device)
    obs_dim = flat_obs.shape[-1]
    action_dim = int(np.prod(env.action_space.shape))
    print(f"[ppo] obs_dim={obs_dim}, action_dim={action_dim}")

    agent = StatePolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_iterations = cfg.total_timesteps // batch_size

    buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, obs_dim, action_dim, device)

    writer = SummaryWriter(str(run_dir))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(cfg).items()))

    global_step = 0
    start_time = time.time()

    next_obs = obs["policy"]
    next_done = torch.zeros(cfg.num_envs, device=device)

    best_eval_reward = -float("inf")

    for iteration in range(1, num_iterations + 1):
        final_values = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        agent.eval()

        if iteration % cfg.eval_freq == 1:
            print(f"[ppo] iter {iteration} / {num_iterations}, evaluating...")
            # quick eval: just measure rollout reward
            eval_env = env
            obs_eval, _ = eval_env.reset()
            ep_rewards = []
            ep_reward = torch.zeros(cfg.num_envs, device=device)
            for _ in range(env.max_episode_length + 1):
                flat = flatten_obs(obs_eval["policy"], device)
                with torch.no_grad():
                    a = agent.get_action_and_value(flat)[0]
                obs_eval, r, term, trunc, _ = eval_env.step(a)
                ep_reward += r
                done = (term | trunc).bool()
                if done.any():
                    for k in range(done.shape[0]):
                        if done[k]:
                            ep_rewards.append(ep_reward[k].item())
                            ep_reward[k] = 0
                    if len(ep_rewards) >= 5:
                        break
            mean_eval = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            writer.add_scalar("eval/mean_episode_reward", mean_eval, global_step)
            print(f"[ppo] eval mean_episode_reward={mean_eval:.3f}")
            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                torch.save(agent.state_dict(), run_dir / "best_ckpt.pt")
                print(f"[ppo] new best saved (reward={best_eval_reward:.3f})")

        if iteration % cfg.save_freq == 0:
            torch.save(agent.state_dict(), run_dir / f"ckpt_{iteration}.pt")

        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        rollout_start = time.perf_counter()
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            flat = flatten_obs(next_obs, device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(flat)
                value = value.flatten()
            buffer.obs[step] = flat
            buffer.actions[step] = action
            buffer.logprobs[step] = logprob
            buffer.dones[step] = next_done
            buffer.values[step] = value

            obs, reward, terminated, truncated, info = env.step(action)
            if isinstance(reward, (float, int)):
                reward = torch.full((cfg.num_envs,), float(reward), device=device)
            buffer.rewards[step] = reward.view(-1).float()
            next_obs = obs["policy"]
            next_done = (terminated | truncated).float()

        rollout_time = time.perf_counter() - rollout_start

        with torch.no_grad():
            next_flat = flatten_obs(next_obs, device)
            next_value = agent.get_value(next_flat).reshape(1, -1)
            advantages = torch.zeros_like(buffer.rewards)
            lastgae = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnotdone = 1.0 - next_done
                    nextvals = next_value
                else:
                    nextnotdone = 1.0 - buffer.dones[t + 1]
                    nextvals = buffer.values[t + 1]
                real_next = nextnotdone * nextvals + final_values[t]
                delta = buffer.rewards[t] + cfg.gamma * real_next - buffer.values[t]
                advantages[t] = lastgae = delta + cfg.gamma * cfg.gae_lambda * nextnotdone * lastgae
            returns = advantages + buffer.values

        b_obs = buffer.obs.reshape((-1, obs_dim))
        b_logprobs = buffer.logprobs.reshape(-1)
        b_actions = buffer.actions.reshape((-1, action_dim))
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = buffer.values.reshape(-1)

        agent.train()
        update_start = time.perf_counter()
        clipfracs = []
        inds = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start:start + minibatch_size]
                _, newlogp, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = newlogp - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break
        update_time = time.perf_counter() - update_start

        sps = int(global_step / (time.time() - start_time + 1e-8))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout_s", rollout_time, global_step)
        writer.add_scalar("time/update_s", update_time, global_step)

        print(f"[ppo] iter {iteration}/{num_iterations} step={global_step} "
              f"pg_loss={pg_loss.item():.4f} v_loss={v_loss.item():.4f} "
              f"kl={approx_kl.item():.4f} SPS={sps}")

    torch.save(agent.state_dict(), run_dir / "final_ckpt.pt")
    print(f"[ppo] training done, final ckpt saved to {run_dir / 'final_ckpt.pt'}")
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    cfg = PPOConfig(
        task=args_cli.task,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        total_timesteps=args_cli.total_timesteps,
        learning_rate=args_cli.learning_rate,
        num_steps=args_cli.num_steps,
        num_minibatches=args_cli.num_minibatches,
        update_epochs=args_cli.update_epochs,
        gamma=args_cli.gamma,
        gae_lambda=args_cli.gae_lambda,
        clip_coef=args_cli.clip_coef,
        ent_coef=args_cli.ent_coef,
        vf_coef=args_cli.vf_coef,
        max_grad_norm=args_cli.max_grad_norm,
        target_kl=args_cli.target_kl,
        norm_adv=args_cli.norm_adv,
        anneal_lr=args_cli.anneal_lr,
        exp_name=args_cli.exp_name,
        save_freq=args_cli.save_freq,
        eval_freq=args_cli.eval_freq,
        device=args_cli.device,
        headless=args_cli.headless,
    )
    train(cfg)
