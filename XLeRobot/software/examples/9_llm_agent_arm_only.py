#!/usr/bin/env python3
"""
XLeRobot LLM Agent - 仅机械臂版本（无轮子）
使用 SO100Follower 控制机械臂
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件
load_dotenv()

print("="*60)
print("XLeRobot LLM Agent - 仅机械臂版本")
print("="*60)
print()

# ========== 配置硬件 ==========
print("正在初始化硬件...")

robot = None
try:
    from lerobot.robots.so_follower.so_follower import SO100Follower
    from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
    
    # 配置机器人（左臂端口）
    port = "/dev/ttyACM0"
    robot_config = SO100FollowerConfig(port=port)
    robot = SO100Follower(robot_config)
    robot.connect()
    print("✅ 机械臂已连接")
except Exception as e:
    print(f"⚠️  机械臂连接失败: {e}")
    print("   将运行无硬件的模拟模式")

print()

# ========== 初始化 LLM（火山引擎）==========
print("正在初始化 LLM...")
llm = ChatOpenAI(
    model=os.getenv("ARK_MODEL_ID"),
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
    temperature=0.7,
)
print("✅ LLM 已初始化")
print()

# ========== 辅助函数 ==========
def move_arm_to_positions(robot, target_positions, duration=2.0):
    """移动机械臂到目标位置"""
    if not robot:
        print(f"[模拟] 移动到: {target_positions}")
        return
    
    control_freq = 50
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    current_obs = robot.get_observation()
    start_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            start_positions[motor_name] = value
    
    for step in range(total_steps):
        progress = step / total_steps
        interpolated = {}
        
        for joint_name in target_positions:
            if joint_name in start_positions:
                start = start_positions[joint_name]
                end = target_positions[joint_name]
                interpolated[joint_name] = start + (end - start) * progress
        
        robot_action = {}
        for joint_name, pos in interpolated.items():
            robot_action[f"{joint_name}.pos"] = pos
        
        if robot_action:
            robot.send_action(robot_action)
        
        time.sleep(step_time)


def wave_arm():
    """挥手动作"""
    if not robot:
        print("[模拟] 挥手")
        return
    
    wave_positions1 = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 30.0,
        'elbow_flex': 120.0,
        'wrist_flex': 45.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    wave_positions2 = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 60.0,
        'elbow_flex': 120.0,
        'wrist_flex': 45.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 45.0,
        'elbow_flex': 90.0,
        'wrist_flex': 45.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    move_arm_to_positions(robot, wave_positions1, duration=0.5)
    time.sleep(0.3)
    move_arm_to_positions(robot, wave_positions2, duration=0.3)
    time.sleep(0.3)
    move_arm_to_positions(robot, wave_positions1, duration=0.3)
    time.sleep(0.3)
    move_arm_to_positions(robot, zero_positions, duration=0.5)


def open_gripper():
    """打开夹爪"""
    if not robot:
        print("[模拟] 打开夹爪")
        return
    
    robot_action = {'gripper.pos': 90.0}
    robot.send_action(robot_action)
    time.sleep(0.5)


def close_gripper():
    """关闭夹爪"""
    if not robot:
        print("[模拟] 关闭夹爪")
        return
    
    robot_action = {'gripper.pos': 0.0}
    robot.send_action(robot_action)
    time.sleep(0.5)


def go_to_zero():
    """回到零位"""
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 45.0,
        'elbow_flex': 90.0,
        'wrist_flex': 45.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    if robot:
        move_arm_to_positions(robot, zero_positions, duration=2.0)
    else:
        print(f"[模拟] 回到零位: {zero_positions}")


# ========== 交互循环 ==========
print("="*60)
print("🚀 开始交互！")
print("="*60)
print("提示：输入 '退出' 或 'quit' 退出程序")
print("可用命令：")
if robot:
    print("  - '挥手' - 机械臂做挥手动作")
    print("  - '打开夹爪'")
    print("  - '关闭夹爪'")
    print("  - '回到零位'")
else:
    print("  - 无硬件模拟模式")
print()

while True:
    try:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break
        
        print("\n机器人正在思考...")
        
        response = ""
        user_input_lower = user_input.lower()
        
        if "挥手" in user_input_lower or "wave" in user_input_lower:
            print("正在挥手...")
            wave_arm()
            response = "好的！挥手完成"
        elif "打开" in user_input_lower:
            print("正在打开夹爪...")
            open_gripper()
            response = "好的！夹爪已打开"
        elif "关闭" in user_input_lower:
            print("正在关闭夹爪...")
            close_gripper()
            response = "好的！夹爪已关闭"
        elif "零位" in user_input_lower or "回" in user_input_lower:
            print("正在回到零位...")
            go_to_zero()
            response = "好的！已回到零位"
        else:
            # 普通对话
            messages = [
                ("system", "你是一个可以控制 XLeRobot 机械臂的智能助手。你可以挥手、打开夹爪、关闭夹爪、回到零位。"),
                ("human", user_input)
            ]
            ai_response = llm.invoke(messages)
            response = ai_response.content
        
        print(f"\n机器人: {response}\n")
        
    except KeyboardInterrupt:
        print("\n\n再见！")
        break
    except Exception as e:
        print(f"\n出错了: {e}\n")

# 断开连接
if robot:
    try:
        robot.disconnect()
    except:
        pass
