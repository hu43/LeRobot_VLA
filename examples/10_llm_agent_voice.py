#!/usr/bin/env python3
"""
XLeRobot LLM Agent - 语音控制完整示例
"""

from robocrew.core.camera import RobotCamera
from robocrew.core.LLMAgent import LLMAgent
from robocrew.robots.XLeRobot.tools import (
    create_vla_single_arm_manipulation,
    create_go_to_precision_mode,
    create_go_to_normal_mode,
    create_move_backward,
    create_move_forward,
    create_strafe_right,
    create_strafe_left,
    create_look_around,
    create_turn_right,
    create_turn_left
)
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
move_backward = create_move_backward(servo_controler)
turn_left = create_turn_left(servo_controler)
turn_right = create_turn_right(servo_controler)
strafe_left = create_strafe_left(servo_controler)
strafe_right = create_strafe_right(servo_controler)
look_around = create_look_around(servo_controler, main_camera)
go_to_precision_mode = create_go_to_precision_mode(servo_controler)
go_to_normal_mode = create_go_to_normal_mode(servo_controler)

# ========== 创建 VLA 操作工具（需要先训练策略）==========
# 取消下面的注释来启用 VLA 操作
"""
pick_up_object = create_vla_single_arm_manipulation(
    tool_name="Grab_object",
    tool_description="Grab an object from the table.",
    task_prompt="Grab the object.",
    server_address="0.0.0.0:8080",
    policy_name="your_policy_name_here",
    policy_type="act",
    arm_port=right_arm_wheel_usb,
    servo_controler=servo_controler,
    camera_config={
        "main": {"index_or_path": "/dev/video0"},
        "right_arm": {"index_or_path": "/dev/video1"}
    },
    main_camera_object=main_camera,
    policy_device="cpu",
)
"""

# ========== 初始化 Agent（含语音）==========
agent = LLMAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[
        move_forward,
        move_backward,
        strafe_left,
        strafe_right,
        turn_left,
        turn_right,
        look_around,
        go_to_precision_mode,
        go_to_normal_mode,
        # pick_up_object,  # 取消注释来启用
    ],
    history_len=8,
    main_camera=main_camera,
    camera_fov=90,
    servo_controler=servo_controler,
    sounddevice_index=2,     # 麦克风设备索引（根据实际情况修改）
    wakeword="hey robot",    # 唤醒词
    tts=True,                # 启用语音回复
    debug_mode=True,
)

# ========== 设置任务 ==========
agent.task = "Wait for voice commands and execute them."

# ========== 运行 ==========
if __name__ == "__main__":
    print("="*60)
    print("XLeRobot LLM Agent (语音控制) 启动")
    print("="*60)
    print("说 'hey robot' 来唤醒机器人")
    print()
    agent.go()
