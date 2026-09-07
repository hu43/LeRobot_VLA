# XLeRobot LLM Agent（火山引擎版本）

## 📋 当前硬件状态

根据你的配置：
- ✅ **机械臂** - 已连接
- ❌ **轮子** - 未连接
- ✅ **摄像头** - 已连接

## 🎯 可以测试的功能

### 1. 纯 LLM 对话测试（无需任何硬件）
```bash
conda activate robocrew
cd /home/zhoulin/public_project/XLeRobot/software/examples
python test_llm_ark_generic.py
```

### 2. 仅机械臂控制（推荐！）
```bash
python 9_llm_agent_arm_only.py
```
这个脚本：
- ✅ 跳过轮子初始化（只连接左臂/头部）
- ✅ 提供机械臂控制工具
- ✅ 如果硬件连接失败，自动进入模拟模式
- 🛠️ 可用命令：
  - "挥手" - 机械臂做挥手动作
  - "打开夹爪"
  - "关闭夹爪"
  - "回到零位"

### 3. 完整版本（需要轮子）
```bash
python 9_llm_agent_simple_ark_complete.py
```
⚠️ 当前轮子未连接，运行此脚本会报错

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `test_llm_ark_generic.py` | 🔥 推荐先试！只测试 LLM |
| `9_llm_agent_arm_only.py` | 🔥 推荐！仅机械臂版本 |
| `9_llm_agent_simple_ark_complete.py` | 完整版本（需要轮子） |
| `verify_llm_agent.py` | 环境验证 |
| `.env` | 配置文件 |

## 🚀 快速开始

1. **验证环境：**
```bash
python verify_llm_agent.py
```

2. **测试 LLM：**
```bash
python test_llm_ark_generic.py
```

3. **控制机械臂：**
```bash
python 9_llm_agent_arm_only.py
```
