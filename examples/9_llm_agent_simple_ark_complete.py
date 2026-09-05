#!/usr/bin/env python3
"""
XLeRobot LLM Agent - 完整版本（使用火山引擎）
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from robocrew.core.camera import RobotCamera
from robocrew.robots.XLeRobot.servo_controls import ServoControler

# 加载 .env 文件
load_dotenv()

# ========== 配置硬件 ==========
# 摄像头（根据实际情况修改）
main_camera = RobotCamera("/dev/video0")

# 舵机控制器（根据实际情况修改端口）
right_arm_wheel_usb = "/dev/ttyACM1"    # 右臂/轮子端口
left_arm_head_usb = "/dev/ttyACM0"      # 左臂/头部端口
servo_controler = ServoControler(right_arm_wheel_usb, left_arm_head_usb)

# ========== 创建移动工具 ==========
@tool
def move_forward(duration: float = 1.0):
    """机器人向前移动，duration 为移动时间（秒）"""
    servo_controler.set_wheel_speeds(0.5, 0.5)
    time.sleep(duration)
    servo_controler.set_wheel_speeds(0, 0)
    return f"向前移动 {duration} 秒完成"

@tool
def turn_left(duration: float = 0.5):
    """机器人向左转，duration 为转动时间（秒）"""
    servo_controler.set_wheel_speeds(-0.3, 0.3)
    time.sleep(duration)
    servo_controler.set_wheel_speeds(0, 0)
    return f"左转 {duration} 秒完成"

@tool
def turn_right(duration: float = 0.5):
    """机器人向右转，duration 为转动时间（秒）"""
    servo_controler.set_wheel_speeds(0.3, -0.3)
    time.sleep(duration)
    servo_controler.set_wheel_speeds(0, 0)
    return f"右转 {duration} 秒完成"

@tool
def stop():
    """停止机器人所有运动"""
    servo_controler.set_wheel_speeds(0, 0)
    return "已停止"

tools = [move_forward, turn_left, turn_right, stop]

# ========== 初始化 LLM（火山引擎）==========
llm = ChatOpenAI(
    model=os.getenv("ARK_MODEL_ID"),
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
    temperature=0.7,
)

# ========== 创建 Agent ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个可以控制 XLeRobot 机器人的智能助手。你可以调用工具来移动机器人。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ========== 运行 ==========
if __name__ == "__main__":
    print("="*60)
    print("XLeRobot LLM Agent 启动（使用火山引擎）")
    print("="*60)
    print()
    
    # 简单交互循环
    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("再见！")
                break
            
            print("机器人正在思考...")
            result = agent_executor.invoke({"input": user_input})
            print(f"机器人: {result['output']}")
            print()
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"出错了: {e}")
