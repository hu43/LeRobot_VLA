# Squint — SO-101 Sim-to-Real Cube Grasping

> Fork of [aalmuzairee/squint](https://github.com/aalmuzairee/squint) with fixes for SO-101 deployment on dual-GPU setups and wrist_roll motor bypass.

## Modifications from Upstream

### Multi-GPU Fix
- **envs/lift.py, reach.py, place.py, stack.py**: Added explicit `device=self.device` to all tensor creation and `env_idx.to(self.device)` to fix `cuda:0`/`cuda:1` mismatch on dual-GPU machines.

### Motor 5 (wrist_roll) Bypass
The wrist_roll motor (motor 5) has mechanical friction causing overload. Fixes in:
- **deploy_utils/robot_config.py**: Removes `wrist_roll` from `bus.motors` before connect, invalidates `ids` cache, monkey-patches `write_calibration` to skip missing motors.
- **deploy_utils/manipulator.py**: Individual motor reads (no `sync_read`), returns 0.0 for disabled motors, drops writes to disabled/overloaded motors, overload detection with auto-disable.

### Camera Alignment (Tuned for Our Setup)
- **envs/base_random_env.py**: Wrist camera parameters tuned via `deploy_utils/tune_camera.py`:
  ```python
  WRIST_CAMERA_BASE_POS = (-0.0120, 0.1040, -0.0010)
  WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-98.0), np.deg2rad(85.0), np.deg2rad(-33.0))
  WRIST_CAMERA_FOV = np.deg2rad(71.0)
  ```

## Quick Start

### Environment
```bash
conda activate hupy3.11   # training
conda activate hupy3.12   # deployment
```

### Training
```bash
cd /home/zhoulin/hu/lerobot/rl/squint
CUDA_VISIBLE_DEVICES=1 python train_squint.py \
    --env_id=SO101LiftCube-v1 \
    --total_timesteps=1500000 \
    --no-cudagraphs
```
~30 min on RTX 5090. Checkpoint saved to `runs/baseline/ckpt.pt`.

### Camera Alignment
```bash
CUDA_VISIBLE_DEVICES=1 python deploy_utils/tune_camera.py --env-id SO101LiftCube-v1
```
Align **static elements** (table, robot arm, background) in the Blended view — NOT the cube. Press `p` to print params, then copy to `envs/base_random_env.py`.

### Deploy to Real Robot
```bash
CUDA_VISIBLE_DEVICES=1 python deploy.py \
    --checkpoint=runs/baseline/ckpt.pt \
    --env-id=SO101LiftCube-v1 \
    --debug
```
- `s` — skip episode
- `q` — quit
- `--no-continuous_eval` — step-by-step mode for safety

### Calibration
Uses `deploy_utils/zihao_follower_arm.json` (SO-101 calibration file).

---

# Squint (Original README)

<p align="center">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/reach_cube.gif">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/reach_can.gif">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/lift_cube.gif">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/lift_can.gif">
<br>
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/place_cube.gif">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/place_can.gif">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/stack_cube.gif">
<img width="24%" src="https://github.com/aalmuzairee/squint/blob/gh-pages/static/extras/gifs/stack_can.gif">
</p>

**Fast Visual Reinforcement Learning for Sim-to-Real Robotics**

Squint is a visual Soft Actor Critic method, that through careful image preprocessing, architectural design choices, and hyperparameter selection, is able to leverage parallel environments and experience reuse effectively, achieving faster wall-clock training time than both prior visual off-policy and on-policy methods, and *solving visual tasks in minutes*.

Pytorch Implementation for [[Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics]](https://arxiv.org/abs/2602.21203) by

[Abdulaziz Almuzairee](https://aalmuzairee.github.io) and [Henrik I. Christensen](https://hichristensen.com) (UC San Diego)</br>

[[Website]](https://aalmuzairee.github.io/squint) [[Paper]](https://arxiv.org/abs/2602.21203)

If you use this code in your research, kindly cite:

```bibtex
@article{almuzairee2026squint,
      title={Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics},
      author={Almuzairee, Abdulaziz and Christensen, Henrik I.},
      journal={arXiv preprint arXiv:2602.21203},
      year={2026}
}
```

-----

## Requirements

- **GPU**: NVIDIA RTX 3080 or better (At least 10GB GPU RAM)
- **Robot**: SO-101 robot arm and wrist camera

## Installation

```bash
conda env create -f environment.yaml
conda activate squint
```

## Simulation Training

```bash
python train_squint.py --env_id=SO101LiftCube-v1
```

### Available Environments (SO-101 Task Set)

| Environment | Description | Time to Convergence |
|-------------|-------------|---------------------|
| `SO101ReachCube-v1` | Reach to a target cube position | 2 minutes |
| `SO101ReachCan-v1` | Reach to a target can position | 2 minutes |
| `SO101LiftCube-v1` | Pick up and lift a cube | 3 minutes |
| `SO101LiftCan-v1` | Pick up and lift a can | 4 minutes |
| `SO101PlaceCube-v1` | Pick up a cube and place in the bin | 5 minutes |
| `SO101PlaceCan-v1` | Pick up a can and place in the bin | 6 minutes |
| `SO101StackCube-v1` | Stack the smaller cube on the larger one | 6 minutes |
| `SO101StackCan-v1` | Stack the cube on the can | 9 minutes |

## Deployment on Real SO-101 Robot

### Step 1: Configure Your Robot
Edit `deploy_utils/robot_config.py` with your hardware settings.

### Step 2: Tune Camera Alignment
Align your real camera view with the simulation by matching the **gripper and base positions** (not the objects):

```bash
python deploy_utils/tune_camera.py
```

Press `p` to print wrist camera parameters, then copy them to `envs/base_random_env.py`.

### Step 3: Deploy

```bash
python deploy.py --checkpoint=path/to/ckpt.pt --env_id=SO101LiftCube-v1
```

## Project Structure

```
squint/
├── train_squint.py          # Main training script
├── deploy.py                # Real robot deployment
├── utils.py                 # Training utilities
├── environment.yaml         # Conda environment
├── envs/                    # Custom ManiSkill environments
│   ├── base_random_env.py   # Base env with domain randomization
│   ├── black_overlay.png    # Background overlay for sim-to-real
│   ├── reach.py / lift.py / place.py / stack.py
│   └── robot/               # Robot URDF and meshes
├── examples/
│   └── visualize_sim.py     # Visualize all environments
└── deploy_utils/
    ├── robot_config.py      # Robot hardware config
    ├── manipulator.py       # Real robot interface
    └── tune_camera.py       # Camera alignment tool
```

## Acknowledgments

- [LeanRL](https://github.com/meta-pytorch/LeanRL)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [ManiSkill3](https://github.com/haosulab/ManiSkill)
- [LeRobot Sim2Real ManiSkill3](https://github.com/StoneT2000/lerobot-sim2real)
- [LeRobot](https://github.com/huggingface/lerobot)

## License

This project is [MIT Licensed](LICENSE).
