# XLeRobot LLM Agent（火山引擎版本）

## 🎉 已完成！

- ✅ 环境配置
- ✅ 火山引擎 API 连接成功
- ✅ LLM 测试通过

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `test_llm_ark_generic.py` | 只测试 LLM（不需要硬件） |
| `9_llm_agent_simple_ark.py` | 简单版本 |
| `9_llm_agent_simple_ark_complete.py` | 完整版本（需要机器人硬件） |
| `.env` | 配置文件 |

## 🚀 使用

### 1. 测试 LLM（推荐先试）
```bash
conda activate robocrew
cd /home/zhoulin/public_project/XLeRobot/software/examples
python test_llm_ark_generic.py
```

### 2. 运行完整版本（需要机器人硬件）
```bash
python 9_llm_agent_simple_ark_complete.py
```
