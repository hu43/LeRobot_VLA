# Fold Cloth 任务完整部署指南

本文档详细说明如何实现、训练和部署 Fold Cloth（叠衣服）任务到 SO-101 真机。

## 目录

1. [任务概述](#任务概述)
2. [环境搭建](#环境搭建)
3. [状态机数据生成](#状态机数据生成)
4. [使用预训练模型](#使用预训练模型)
5. [仿真部署验证](#仿真部署验证)
6. [真机部署](#真机部署)

---

## 任务概述

Fold Cloth 是一个双臂操作任务，使用两个 SO-101 机械臂协同将一块布折叠起来。

**任务 ID**: `LeIsaac-SO101-FoldCloth-BiArm-Direct-v0`

**目标**: 左手臂抓住左袖子，右手臂抓住右袖子，同时向中间移动完成折叠，然后回到休息位置。

---

## 环境搭建

### 1. 安装依赖

```bash
cd leisaac

# 安装 leisaac 及其依赖
pip install -e "source/leisaac[all]"

# 安装 Isaac Lab (按照官方指南)
```

### 2. 验证环境注册

```bash
python scripts/environments/list_envs.py
```

确认可以看到 `LeIsaac-SO101-FoldCloth-BiArm-Direct-v0`。

---

## 状态机数据生成

### 1. 使用状态机生成演示数据

我们已经实现了 `FoldClothStateMachine`，可以自动生成演示数据：

```bash
# 生成 50 个演示轨迹
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/fold_cloth.hdf5 \
    --num_demos 50
```

### 2. 回放生成的数据

```bash
# 回放已录制的演示
python scripts/datagen/state_machine/replay.py \
    --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --dataset_file ./datasets/fold_cloth.hdf5
```

---

## 使用预训练模型

### 方法一: 使用 LightwheelAI 官方模型

LightwheelAI 在 HuggingFace 上提供了预训练模型：

| 模型 | HuggingFace 地址 | 任务 |
|------|-----------------|------|
| GR00T N1.5 (Pick Orange) | `LightwheelAI/leisaac-pick-orange-v0` | Pick Orange |
| GR00T N1.5 (Fold Cloth) | `LightwheelAI/leisaac-fold-cloth-v0` | Fold Cloth |

```bash
# 下载模型
git lfs install
git clone https://huggingface.co/LightwheelAI/leisaac-fold-cloth-v0
```

### 方法二: 使用 GR00T 官方模型

NVIDIA GR00T 提供了基础模型，可以微调：

```bash
# 克隆 GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout 4af2b62  # N1.5 版本
```

### 方法三: 从头训练

如果你想从头训练自己的模型：

```bash
# 1. 转换数据到 LeRobot 格式
python scripts/convert/isaaclab2lerobot.py \
    --task_name LeIsaac-SO101-FoldCloth-v0 \
    --repo_id your_hf_username/fold-cloth-dataset \
    --hdf5_root ./datasets \
    --hdf5_files fold_cloth.hdf5

# 2. 使用 LeRobot 训练
# 参见 LeRobot 官方文档
```

---

## 仿真部署验证

### 1. 启动策略服务器 (GR00T)

```bash
cd Isaac-GR00T

# 启动 GR00T N1.5 推理服务器
python scripts/inference_service.py \
    --checkpoint_path /path/to/leisaac-fold-cloth-v0 \
    --host localhost \
    --port 5555
```

### 2. 在仿真中运行推理

```bash
cd leisaac

python scripts/evaluation/policy_inference.py \
    --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --eval_rounds 10 \
    --policy_type gr00tn1.5 \
    --policy_host localhost \
    --policy_port 5555 \
    --policy_action_horizon 16 \
    --policy_language_instruction "Fold the cloth, and reset the arm to rest state." \
    --device cuda \
    --enable_cameras
```

### 3. 验证成功率

运行推理后，检查控制台输出的成功率。目标成功率 > 80%。

---

## 真机部署

### 1. 硬件准备

- 2x SO-101 机械臂 (左臂 + 右臂)
- 2x 相机 (前视图 + 手腕视图)
- RealSense D435 或类似相机
- 折叠布道具

### 2. 软件配置

#### 配置真实机器人

创建或修改 `lerobot-sim2real/lerobot_sim2real/config/real_robot.py`:

```python
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig

def create_real_robot(uid: str = "bi_so101"):
    if uid == "bi_so101":
        # 左臂配置
        left_config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            use_degrees=True,
            cameras={
                "left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="LEFT_CAMERA_SERIAL",
                    fps=30,
                    width=640,
                    height=480
                ),
                "front": RealSenseCameraConfig(
                    serial_number_or_name="FRONT_CAMERA_SERIAL",
                    fps=30,
                    width=640,
                    height=480
                )
            },
            id="left_arm"
        )
        # 右臂配置
        right_config = SO101FollowerConfig(
            port="/dev/ttyACM1",
            use_degrees=True,
            cameras={
                "right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="RIGHT_CAMERA_SERIAL",
                    fps=30,
                    width=640,
                    height=480
                )
            },
            id="right_arm"
        )
        return (make_robot_from_config(left_config), make_robot_from_config(right_config))
```

### 3. 安全设置

```bash
# 设置紧急停止
# 确保在运行前测试机械臂的运动范围
```

### 4. 启动策略服务器

```bash
# 在服务器机器上启动
cd Isaac-GR00T
python scripts/inference_service.py \
    --checkpoint_path /path/to/leisaac-fold-cloth-v0 \
    --host 0.0.0.0 \
    --port 5555
```

### 5. 真机部署运行

```bash
cd lerobot-sim2real

python lerobot_sim2real/scripts/eval_ppo_rgb.py \
    --checkpoint /path/to/leisaac-fold-cloth-v0 \
    --env_id LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --debug \
    --continuous_eval \
    --max_episode_steps 200 \
    --num_episodes 10 \
    --control_freq 15
```

---

## 故障排除

### 常见问题

1. **仿真中布料无法正确加载**
   - 检查 USD 路径是否正确
   - 确保粒子系统配置正确

2. **策略服务器连接失败**
   - 检查防火墙设置
   - 验证主机 IP 和端口配置

3. **真机运动幅度过大**
   - 调整 `control_freq` 到更低的值 (如 10Hz)
   - 减小动作缩放比例

---

## 参考资源

- [LeIsaac 官方文档](https://lightwheelai.github.io/leisaac/)
- [GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [HuggingFace LightwheelAI 组织](https://huggingface.co/LightwheelAI)

---

## 快速开始命令汇总

```bash
# ========== 1. 状态机生成数据 ==========
cd leisaac
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record --dataset_file ./datasets/fold_cloth.hdf5 --num_demos 50

# ========== 2. 转换数据格式 ==========
python scripts/convert/isaaclab2lerobot.py \
    --task_name LeIsaac-SO101-FoldCloth-v0 \
    --repo_id your_hf_username/fold-cloth-dataset \
    --hdf5_root ./datasets --hdf5_files fold_cloth.hdf5

# ========== 3. 启动策略服务器 (GR00T) ==========
cd ../Isaac-GR00T
python scripts/inference_service.py \
    --checkpoint_path /path/to/leisaac-fold-cloth-v0 \
    --host localhost --port 5555

# ========== 4. 仿真验证 ==========
cd ../leisaac
python scripts/evaluation/policy_inference.py \
    --task LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --eval_rounds 10 --policy_type gr00tn1.5 \
    --policy_host localhost --policy_port 5555 \
    --policy_language_instruction "Fold the cloth, and reset the arm to rest state." \
    --device cuda --enable_cameras

# ========== 5. 真机部署 ==========
cd ../lerobot-sim2real
python lerobot_sim2real/scripts/eval_ppo_rgb.py \
    --checkpoint /path/to/leisaac-fold-cloth-v0 \
    --env_id LeIsaac-SO101-FoldCloth-BiArm-Direct-v0 \
    --num_episodes 10 --control_freq 15
```
