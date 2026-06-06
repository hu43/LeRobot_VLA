#!/usr/bin/env python3
"""
XLeRobot LLM Agent - 简单移动示例
"""

from robocrew.core.camera import RobotCamera
from robocrew.core.LLMAgent import LLMAgent
from robocrew.robots.XLeRobot.tools import create_move_forward, create_turn_right, create_turn_left
from robocrew.robots.XLeRobot.servo_controls import ServoControler

# ========== 配置硬件 ==========
# 摄像头（根据实际情况修改）
main_camera = RobotCamera("/dev/video0")

# 舵机控制器（根据实际情况修改端口）
right_arm_wheel_usb = "/dev/ttyACM1"    # 右臂/轮子端口
left_arm_head_usb = "/dev/ttyACM0"      # 左臂/头部端口
servo_controler = ServoControler(right_arm_wheel_usb, left_arm_head_usb)

# ========== 创建移动工具 ==========
move_forward = create_move_forward(servo_controler)
turn_left = create_turn_left(servo_controler)
turn_right = create_turn_right(servo_controler)

# ========== 初始化 Agent ==========
agent = LLMAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[move_forward, turn_left, turn_right],
    main_camera=main_camera,
    servo_controler=servo_controler,
    debug_mode=True,
)

# ========== 设置任务 ==========
agent.task = "Approach a human and stop."

# ========== 运行 ==========
if __name__ == "__main__":
    print("="*60)
    print("XLeRobot LLM Agent 启动")
    print("="*60)
    print(f"任务: {agent.task}")
    print()
    agent.go()
