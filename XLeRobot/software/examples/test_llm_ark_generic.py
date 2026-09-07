#!/usr/bin/env python3
"""
通用测试火山引擎（Ark）LLM 连接
请在 .env 中配置：
- ARK_API_KEY
- ARK_MODEL_ID
- ARK_BASE_URL
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

print("="*60)
print("测试火山引擎（Ark）LLM 连接")
print("="*60)
print()

# 检查配置
api_key = os.getenv("ARK_API_KEY")
model_id = os.getenv("ARK_MODEL_ID")
base_url = os.getenv("ARK_BASE_URL")

if not api_key:
    print("❌ 未找到 ARK_API_KEY")
    exit(1)
if not model_id or model_id == "你的模型ID":
    print("❌ 请在 .env 中配置 ARK_MODEL_ID")
    print("   访问火山引擎控制台获取接入点 ID（格式：ep-xxx...）")
    exit(1)
if not base_url:
    print("❌ 未找到 ARK_BASE_URL")
    exit(1)

print(f"✅ API Key 已加载")
print(f"✅ 模型 ID: {model_id}")
print(f"✅ Base URL: {base_url}")
print()

# 测试火山引擎模型
try:
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )
    
    print("正在测试 LLM...")
    response = llm.invoke("你好，请用一句话介绍你自己")
    print()
    print("="*60)
    print("LLM 响应：")
    print(response.content)
    print("="*60)
    print()
    print("🎉 LLM 测试成功！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("💡 提示：请确认")
    print("  1. ARK_MODEL_ID 正确（接入点 ID，格式：ep-xxx...）")
    print("  2. 火山引擎控制台中该接入点已启用")
    print("  3. API Key 有访问该模型的权限")
