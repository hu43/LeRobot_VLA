# XLeRobot LLM Agent 使用指南

## 📋 已完成的工作

✅ 环境：`robocrew`（Python 3.10）已创建  
✅ RoboCrew 库已安装  
✅ 示例脚本已创建：
- `9_llm_agent_simple.py` - 简单移动示例（Gemini）
- `9_llm_agent_simple_deepseek.py` - 简单移动示例（DeepSeek，推荐）
- `10_llm_agent_voice.py` - 语音控制完整示例
- `.env.example` - API Key 配置模板

## 🚀 快速开始

### 0️⃣ 安装系统依赖（语音功能需要）

```bash
sudo apt install portaudio19-dev
```

### 1️⃣ 配置 API Key

复制模板并填入你的 DeepSeek API Key（推荐国内使用）：

```bash
cd /home/zhoulin/public_project/XLeRobot/software/examples
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY
```

获取 DeepSeek API Key：https://platform.deepseek.com

（可选）获取 Gemini API Key：https://aistudio.google.com/app/apikey

### 2️⃣ 硬件检查（已确认）

| 设备 | 端口 | 状态 |
|------|------|------|
| 右臂/轮子 | `/dev/ttyACM1` | ✅ 已连接 |
| 左臂/头部 | `/dev/ttyACM0` | ✅ 已连接 |
| 摄像头 | `/dev/video0` | ✅ 已连接 |

### 3️⃣ 验证环境（无需硬件）

先运行验证脚本确认环境配置正确：

```bash
conda activate robocrew
python verify_llm_agent.py
```

### 4️⃣ 运行简单示例（需要机器人硬件）

```bash
# 激活环境
conda activate robocrew

# 进入目录
cd /home/zhoulin/public_project/XLeRobot/software/examples

# 运行简单移动示例（DeepSeek，推荐）
python 9_llm_agent_simple_deepseek.py

# 或运行（Gemini 版本）
python 9_llm_agent_simple.py
```

### 5️⃣ 运行语音控制示例（需要麦克风）

```bash
# 先检查麦克风设备
python check_audio_devices.py

# 修改 10_llm_agent_voice.py 里的 sounddevice_index
# 然后运行
python 10_llm_agent_voice.py
```

## 🎯 下一步（可选）

### 启用 VLA 操作

如果你需要机器人手臂操作：

1. 先在 `lerobot` 环境训练 VLA 策略（参考 VLA_ACT 教程）
2. 在另一个终端启动策略服务器：
```bash
conda activate lerobot
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```
3. 在 `10_llm_agent_voice.py` 中取消 VLA 工具的注释

### 固定 USB 端口

防止重启后端口变化：
```bash
conda activate robocrew
robocrew-setup-usb-modules
```

## 📝 脚本说明

| 文件 | 功能 |
|------|------|
| `9_llm_agent_simple.py` | 基础移动，无需语音 |
| `10_llm_agent_voice.py` | 完整功能，含语音控制 |

## ⚙️ 修改配置

编辑脚本里的这些变量：

```python
# 摄像头
main_camera = RobotCamera("/dev/video0")

# 舵机端口
right_arm_wheel_usb = "/dev/ttyACM1"
left_arm_head_usb = "/dev/ttyACM0"

# 麦克风索引
sounddevice_index=2,

# 唤醒词
wakeword="hey robot",
```
