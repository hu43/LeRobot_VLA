#!/usr/bin/env python3
"""
XLeRobot - VLA 模型可视化仿真
使用 jliu6718/lerobot-so101-act 模型进行单机械臂 VLA 推理
并通过 OpenCV 显示摄像头画面和机械臂状态
"""

import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file

import cv2
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType, PolicyFeature

# ===== 配置 =====
MODEL_PATH = "/home/zhoulin/public_project/models/jliu6718/lerobot-so101-act"
DEVICE = "cpu"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS = 10  # 仿真帧率
MAX_STEPS = 300  # 最大仿真步数

# 关节名称（SO101 单臂 6 自由度）
JOINT_NAMES = [
    "shoulder_pan",   # 肩部水平旋转
    "shoulder_lift",  # 肩部垂直抬起
    "elbow_flex",     # 肘部弯曲
    "wrist_flex",     # 腕部弯曲
    "wrist_roll",     # 腕部旋转
    "gripper",        # 夹爪开合
]

# 关节角度范围（归一化后约为 [-1, 1]，真实角度单位：度）
JOINT_RANGES = [
    (-180, 180),   # shoulder_pan
    (-90, 90),     # shoulder_lift
    (-180, 180),   # elbow_flex
    (-90, 90),     # wrist_flex
    (-180, 180),   # wrist_roll
    (0, 90),       # gripper
]

# 连杆长度（像素单位，用于 2D 可视化）
LINK_LENGTHS = [60, 80, 60, 40, 30, 20]


# ===== 工具函数 =====
def load_vla_policy():
    """加载 VLA (ACT) 策略"""
    print("正在加载 VLA 策略...")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  设备: {DEVICE}")

    with open(f"{MODEL_PATH}/config.json", "r") as f:
        cfg = json.load(f)

    act_cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        chunk_size=cfg["chunk_size"],
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        vision_backbone=cfg.get("vision_backbone", "resnet18"),
        dim_model=cfg.get("dim_model", 512),
        n_heads=cfg.get("n_heads", 8),
        n_encoder_layers=cfg.get("n_encoder_layers", 4),
        n_decoder_layers=cfg.get("n_decoder_layers", 1),
        use_vae=cfg.get("use_vae", True),
        latent_dim=cfg.get("latent_dim", 32),
        temporal_ensemble_coeff=cfg.get("temporal_ensemble_coeff", None),
        device=DEVICE,
    )

    policy = ACTPolicy(act_cfg)
    policy.eval()

    state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    policy.to(DEVICE)
    policy.eval()

    print(f"✅ 策略加载完成！参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"   missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return policy


def create_simulated_scene(step, arm_state):
    """创建模拟的桌面场景图像（作为"摄像头输入"）"""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # 桌面背景（渐变）
    for y in range(IMAGE_HEIGHT):
        color = int(80 + (y / IMAGE_HEIGHT) * 40)
        image[y, :] = (color, color, color + 20)

    # 绘制一个立方体（目标物体）
    cube_x = IMAGE_WIDTH // 2 + int(np.sin(step * 0.05) * 80)
    cube_y = IMAGE_HEIGHT // 2 + 50
    cube_size = 60

    cv2.rectangle(
        image,
        (cube_x - cube_size // 2, cube_y - cube_size // 2),
        (cube_x + cube_size // 2, cube_y + cube_size // 2),
        (200, 50, 50),
        -1,
    )
    # 立方体立体效果
    cv2.line(image, (cube_x - cube_size // 2, cube_y - cube_size // 2),
             (cube_x - cube_size // 2 + 20, cube_y - cube_size // 2 - 20), (150, 30, 30), 2)
    cv2.line(image, (cube_x + cube_size // 2, cube_y - cube_size // 2),
             (cube_x + cube_size // 2 + 20, cube_y - cube_size // 2 - 20), (150, 30, 30), 2)
    cv2.line(image, (cube_x - cube_size // 2 + 20, cube_y - cube_size // 2 - 20),
             (cube_x + cube_size // 2 + 20, cube_y - cube_size // 2 - 20), (150, 30, 30), 2)
    cv2.line(image, (cube_x + cube_size // 2, cube_y - cube_size // 2),
             (cube_x + cube_size // 2 + 20, cube_y - cube_size // 2 - 20), (150, 30, 30), 2)
    cv2.line(image, (cube_x + cube_size // 2, cube_y + cube_size // 2),
             (cube_x + cube_size // 2 + 20, cube_y + cube_size // 2 - 20), (150, 30, 30), 2)

    # 绘制一个球
    ball_x = IMAGE_WIDTH // 3
    ball_y = IMAGE_HEIGHT // 3
    cv2.circle(image, (ball_x, ball_y), 40, (0, 128, 255), -1)
    cv2.circle(image, (ball_x - 10, ball_y - 10), 10, (200, 220, 255), -1)

    # 绘制桌面边界
    cv2.rectangle(image, (20, 20), (IMAGE_WIDTH - 20, IMAGE_HEIGHT - 20), (100, 100, 100), 2)

    # 绘制提示文本
    cv2.putText(image, "CAMERA FRONT (SIMULATED)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, f"Step: {step}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(image, "VLA Policy: jliu6718/lerobot-so101-act", (20, IMAGE_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return image, (cube_x, cube_y)


def visualize_arm_2d(joint_angles, cube_pos, action=None):
    """创建机械臂的 2D 可视化图像"""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:] = (20, 20, 30)

    # 机械臂基座位置
    base_x = IMAGE_WIDTH // 2
    base_y = IMAGE_HEIGHT - 80

    # 绘制基座
    cv2.rectangle(image, (base_x - 50, base_y - 20), (base_x + 50, base_y + 40), (100, 100, 100), -1)
    cv2.circle(image, (base_x, base_y), 15, (150, 150, 150), -1)
    cv2.putText(image, "BASE", (base_x - 20, base_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 计算机械臂各关节位置（简化的 2D 投影）
    positions = [(base_x, base_y)]

    # 简化：只使用前 3 个主要关节做 2D 投影
    pan = joint_angles[0]  # shoulder_pan (水平旋转) - 影响 x
    lift = joint_angles[1]  # shoulder_lift (垂直抬起) - 影响 y
    elbow = joint_angles[2]  # elbow_flex - 影响 y
    wrist = joint_angles[3]  # wrist_flex - 影响末端
    roll = joint_angles[4]  # wrist_roll
    gripper = joint_angles[5]  # gripper

    # 第一段：肩 -> 肘
    angle1 = np.radians(90 - lift)  # 抬起角度
    x1 = base_x + int(LINK_LENGTHS[0] * np.cos(angle1) * np.cos(np.radians(pan * 0.3)))
    y1 = base_y - int(LINK_LENGTHS[0] * np.sin(angle1))
    positions.append((x1, y1))
    cv2.line(image, positions[0], positions[1], (0, 200, 255), 5)
    cv2.circle(image, positions[1], 8, (255, 200, 0), -1)

    # 第二段：肘 -> 腕
    angle2 = angle1 + np.radians(elbow - 90)
    x2 = x1 + int(LINK_LENGTHS[1] * np.cos(angle2))
    y2 = y1 - int(LINK_LENGTHS[1] * np.sin(angle2))
    positions.append((x2, y2))
    cv2.line(image, positions[1], positions[2], (0, 200, 255), 5)
    cv2.circle(image, positions[2], 8, (255, 200, 0), -1)

    # 第三段：腕 -> 末端
    angle3 = angle2 + np.radians(wrist - 45)
    x3 = x2 + int(LINK_LENGTHS[2] * np.cos(angle3))
    y3 = y2 - int(LINK_LENGTHS[2] * np.sin(angle3))
    positions.append((x3, y3))
    cv2.line(image, positions[2], positions[3], (0, 200, 255), 5)
    cv2.circle(image, positions[3], 8, (255, 200, 0), -1)

    # 夹爪
    gripper_open = (gripper + 1) * 10  # 归一化值 [-1,1] -> [0, 20]
    if gripper_open < 0:
        gripper_open = 0
    gx1 = x3 + int(LINK_LENGTHS[3] * np.cos(angle3))
    gy1 = y3 - int(LINK_LENGTHS[3] * np.sin(angle3))
    cv2.line(image, positions[3], (gx1, gy1), (255, 100, 100), 4)

    # 夹爪两个指
    finger_angle1 = angle3 - np.radians(10 + gripper_open)
    finger_angle2 = angle3 + np.radians(10 + gripper_open)
    fx1 = gx1 + int(25 * np.cos(finger_angle1))
    fy1 = gy1 - int(25 * np.sin(finger_angle1))
    fx2 = gx1 + int(25 * np.cos(finger_angle2))
    fy2 = gy1 - int(25 * np.sin(finger_angle2))
    cv2.line(image, (gx1, gy1), (fx1, fy1), (255, 150, 150), 3)
    cv2.line(image, (gx1, gy1), (fx2, fy2), (255, 150, 150), 3)

    # 末端执行器位置高亮
    cv2.circle(image, (gx1, gy1), 6, (0, 255, 0), -1)

    # 绘制目标立方体位置（从场景中获得）
    if cube_pos:
        cx, cy = cube_pos
        cv2.circle(image, (cx, cy - 50), 15, (0, 100, 200), 2)  # 调整 y 以匹配视角
        cv2.putText(image, "TARGET", (cx - 30, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # 绘制动作向量（如果有）
    if action is not None:
        action_text = f"Action: [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}, {action[3]:+.2f}, {action[4]:+.2f}, {action[5]:+.2f}]"
        cv2.putText(image, action_text[:50], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        if len(action_text) > 50:
            cv2.putText(image, action_text[50:], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    # 标题
    cv2.putText(image, "ARM 2D VISUALIZATION", (10, IMAGE_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return image, (gx1, gy1)


def create_joint_bar_chart(joint_angles, action):
    """创建关节角度和动作的柱状图"""
    image = np.zeros((300, 900, 3), dtype=np.uint8)
    image[:] = (30, 30, 40)

    bar_width = 100
    gap = 30
    start_x = 50
    max_bar_height = 200
    base_y = 250

    cv2.putText(image, "JOINT STATES (blue) | VLA ACTIONS (red)", (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, (name, state, act) in enumerate(zip(JOINT_NAMES, joint_angles, action)):
        x = start_x + i * (bar_width + gap)

        # 状态柱状图 (蓝色)
        state_h = int(abs(state) * max_bar_height / 2)
        if state >= 0:
            cv2.rectangle(image, (x, base_y - state_h), (x + bar_width, base_y), (200, 150, 50), -1)
        else:
            cv2.rectangle(image, (x, base_y), (x + bar_width, base_y + state_h), (200, 150, 50), -1)

        # 动作柱状图 (红色，叠加显示)
        act_h = int(abs(act) * max_bar_height / 2)
        if act >= 0:
            cv2.rectangle(image, (x + 20, base_y - act_h - 5), (x + bar_width - 20, base_y - 5), (0, 100, 255), -1)
        else:
            cv2.rectangle(image, (x + 20, base_y + 5), (x + bar_width - 20, base_y + act_h + 5), (0, 100, 255), -1)

        # 关节名称
        cv2.putText(image, name, (x, base_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(image, f"st={state:+.2f}", (x, base_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
        cv2.putText(image, f"ac={act:+.2f}", (x, base_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 255), 1)

    # 基准线
    cv2.line(image, (0, base_y), (image.shape[1], base_y), (100, 100, 100), 1)

    return image


def image_to_tensor(image):
    """将 OpenCV BGR 图像转换为 PyTorch 张量 (1, 3, H, W)"""
    # BGR -> RGB, 归一化到 [0, 1]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ===== 主仿真循环 =====
def main():
    print("=" * 70)
    print("XLeRobot VLA 可视化仿真")
    print("=" * 70)
    print()

    # 1. 加载 VLA 策略
    policy = load_vla_policy()
    print()

    # 2. 初始化机械臂状态
    joint_states = np.zeros(6, dtype=np.float32)  # 归一化 [-1, 1] 范围
    print(f"初始关节状态: {joint_states}")
    print()

    # 3. 创建窗口
    cv2.namedWindow("VLA Simulation - Camera Front", cv2.WINDOW_NORMAL)
    cv2.namedWindow("VLA Simulation - Arm State", cv2.WINDOW_NORMAL)
    cv2.namedWindow("VLA Simulation - Joint Actions", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VLA Simulation - Camera Front", 640, 480)
    cv2.resizeWindow("VLA Simulation - Arm State", 640, 480)
    cv2.resizeWindow("VLA Simulation - Joint Actions", 900, 300)

    print("🚀 开始仿真...")
    print("   按 'q' 或 ESC 退出")
    print("   按 'r' 重置机械臂")
    print("   按 ' ' 暂停/继续")
    print()

    paused = False
    action_history = []

    for step in range(MAX_STEPS):
        # 检查按键
        key = cv2.waitKey(1000 // FPS) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            print("退出仿真")
            break
        elif key == ord('r'):
            joint_states = np.zeros(6, dtype=np.float32)
            print("已重置机械臂状态")
        elif key == ord(' '):
            paused = not paused
            print("暂停" if paused else "继续")

        if paused:
            continue

        # 3. 创建模拟场景图像
        scene_image, cube_pos = create_simulated_scene(step, joint_states)

        # 4. 将图像转换为张量并送入 VLA 推理
        image_tensor = image_to_tensor(scene_image)
        state_tensor = torch.from_numpy(joint_states).unsqueeze(0).float()

        batch = {
            "observation.state": state_tensor.to(DEVICE),
            "observation.images.front": image_tensor.to(DEVICE),
        }

        # 5. VLA 推理
        with torch.no_grad():
            action = policy.select_action(batch)
        action_np = action[0].cpu().numpy()

        # 记录动作历史
        action_history.append(action_np.copy())

        # 6. 应用动作到机械臂状态（简单积分）
        joint_states += action_np * 0.1  # 小步长积分
        joint_states = np.clip(joint_states, -1.0, 1.0)  # 限制范围

        # 7. 可视化
        arm_image, end_effector_pos = visualize_arm_2d(joint_states, cube_pos, action_np)
        bar_image = create_joint_bar_chart(joint_states, action_np)

        # 8. 显示图像
        cv2.imshow("VLA Simulation - Camera Front", scene_image)
        cv2.imshow("VLA Simulation - Arm State", arm_image)
        cv2.imshow("VLA Simulation - Joint Actions", bar_image)

        # 9. 控制台输出（每 10 步一次）
        if step % 10 == 0:
            print(f"[Step {step:3d}] State=[{joint_states[0]:+.2f},{joint_states[1]:+.2f},"
                  f"{joint_states[2]:+.2f},{joint_states[3]:+.2f},{joint_states[4]:+.2f},"
                  f"{joint_states[5]:+.2f}]")

    # 10. 总结
    print()
    print("=" * 70)
    print("✅ 仿真完成！")
    print(f"   总步数: {len(action_history)}")
    if action_history:
        action_array = np.array(action_history)
        print(f"   平均动作绝对值: {np.mean(np.abs(action_array), axis=0)}")
        print(f"   最大动作值: {np.max(action_array, axis=0)}")
        print(f"   最小动作值: {np.min(action_array, axis=0)}")
    print("=" * 70)

    # 等待用户按键关闭窗口
    print()
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
