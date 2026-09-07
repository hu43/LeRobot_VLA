#!/usr/bin/env python3
"""
验证 LLM Agent 环境配置（无需硬件）
"""

import importlib

print("="*60)
print("XLeRobot LLM Agent 环境验证（火山引擎版本）")
print("="*60)
print()

# 检查关键依赖
dependencies = [
    ("robocrew", "RoboCrew 核心库"),
    ("langchain", "LangChain"),
    ("langchain_openai", "LangChain OpenAI"),
    ("torch", "PyTorch"),
]

all_ok = True
for dep_name, desc in dependencies:
    try:
        lib = importlib.import_module(dep_name)
        if hasattr(lib, "__version__"):
            print(f"✅ {desc}: {lib.__version__}")
        else:
            print(f"✅ {desc}")
    except ImportError:
        print(f"❌ {desc}: 未安装")
        all_ok = False

print()

# 检查 .env 文件
import os
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    print("✅ .env 文件存在")
    with open(env_path, "r") as f:
        content = f.read()
    # 检查火山引擎配置
    if "ARK_API_KEY" in content:
        if "your_api_key_here" not in content and "ark-" in content:
            print("✅ ARK_API_KEY 已配置")
        else:
            print("⚠️  请在 .env 中填入你的实际 ARK_API_KEY")
    else:
        print("❌ .env 中没有找到 ARK_API_KEY")
        all_ok = False
    if "ARK_MODEL_ID" in content:
        if "你的模型ID" not in content and "ep-" in content:
            print("✅ ARK_MODEL_ID 已配置")
        else:
            print("⚠️  请在 .env 中填入你的实际 ARK_MODEL_ID")
    else:
        print("❌ .env 中没有找到 ARK_MODEL_ID")
        all_ok = False
    if "ARK_BASE_URL" in content:
        print("✅ ARK_BASE_URL 已配置")
else:
    print("❌ .env 文件不存在，请复制 .env.example 并重命名为 .env")
    all_ok = False

print()

# 检查 RoboCrew 组件
try:
    from robocrew.core.LLMAgent import LLMAgent
    print("✅ LLMAgent 导入成功")
    
    from robocrew.robots.XLeRobot.servo_controls import ServoControler
    print("✅ ServoControler 导入成功")
    
    from robocrew.robots.XLeRobot.tools import create_move_forward
    print("✅ 工具函数导入成功")
except Exception as e:
    print(f"❌ RoboCrew 组件导入失败: {e}")
    all_ok = False

print()
print("="*60)
if all_ok:
    print("🎉 环境配置完成！可以使用 LLM Agent 了")
else:
    print("⚠️  环境配置有问题，请检查上面的错误信息")
print("="*60)
