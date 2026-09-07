#!/usr/bin/env python3
"""
XLeRobot LLM Agent - 纯对话版本（无硬件控制）
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件
load_dotenv()

print("=" * 60)
print("XLeRobot LLM Agent - 纯对话版本")
print("=" * 60)
print()

# ========== 初始化 LLM（火山引擎）==========
print("正在初始化 LLM...")
llm = ChatOpenAI(
    model=os.getenv("ARK_MODEL_ID"),
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
    temperature=0.7,
)
print("LLM 已初始化")
print()

# ========== 交互循环 ==========
print("=" * 60)
print("开始交互！")
print("=" * 60)
print("提示：输入 '退出' 或 'quit' 退出程序")
print()

# 启动问候
print("机器人: 你好，我是朝野的小助手，请问我能为你做什么\n")

while True:
    try:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break

        messages = [
            ("system", "你是朝野的小助手，一个友好、专业的智能助手。请用简洁的中文回答用户的问题。"),
            ("human", user_input)
        ]
        ai_response = llm.invoke(messages)
        print(f"\n机器人: {ai_response.content}\n")

    except KeyboardInterrupt:
        print("\n\n再见！")
        break
    except Exception as e:
        print(f"\n出错了: {e}\n")
