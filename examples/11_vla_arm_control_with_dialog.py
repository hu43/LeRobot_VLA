#!/usr/bin/env python3
"""
朝野小助手 - Pi0 VLA 机械臂控制器
- 真实 SO101 机械臂控制
- 真实摄像头输入
- Pi0 VLA 握手任务推理
- 终端对话交互
"""

import json
import os
import time
import threading
from types import MethodType

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from safetensors.torch import load_file
from ultralytics import YOLO
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL_PATH = "/home/zhoulin/Tang/pi0/50K/pretrained_model"
PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
VLA_MAX_STEPS = 1000
VLA_ACTION_SMOOTH = 0.3
VLA_STABLE_THRESHOLD = 0.3
VLA_STABLE_STEPS = 30

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _detect_device():
    if torch.cuda.is_available():
        try:
            test = torch.randn(1, 3, 224, 224, device="cuda")
            torch.nn.Conv2d(3, 16, 3).to("cuda")(test)
            return "cuda"
        except Exception:
            pass
    return "cpu"


DEVICE = _detect_device()


def load_pi0_policy():
    print(f"  🧠 正在加载 Pi0 VLA 模型 [设备: {DEVICE}]...")

    try:
        __import__("transformers.models.siglip.check")
    except (ImportError, ModuleNotFoundError):
        import sys
        import types

        check_module = types.ModuleType("transformers.models.siglip.check")
        check_module.check_whether_transformers_replace_is_installed_correctly = lambda: True
        sys.modules["transformers.models.siglip.check"] = check_module

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到 Pi0 模型: {MODEL_PATH}")

    with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
        cfg = json.load(f)

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from transformers import AutoTokenizer

    pi0_cfg = PI0Config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        chunk_size=cfg["chunk_size"],
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        pretrained_path=cfg.get("pretrained_path", "lerobot/pi0_base"),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype="float32",
        device=DEVICE,
    )

    policy = PI0Policy(pi0_cfg)
    state_dict = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    policy.to(device=DEVICE, dtype=torch.float32)
    patch_pi0_cache_clone(policy)
    policy.eval()
    policy.reset()

    pre_state = load_file(os.path.join(MODEL_PATH, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_state = load_file(os.path.join(MODEL_PATH, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    norm_params = {
        "state_mean": pre_state["observation.state.mean"].numpy(),
        "state_std": pre_state["observation.state.std"].numpy(),
        "action_mean": post_state["action.mean"].numpy(),
        "action_std": post_state["action.std"].numpy(),
    }

    tokenizer = AutoTokenizer.from_pretrained(PALIGEMMA_TOKENIZER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"    参数: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    print("  ✅ Pi0 VLA 模型加载完成 - 握手任务就绪")
    return policy, tokenizer, norm_params, pi0_cfg.tokenizer_max_length


def pi0_image_to_tensor(image):
    resized = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)


def normalize_state(state_array, norm_params):
    return (state_array - norm_params["state_mean"]) / norm_params["state_std"]


def unnormalize_action(action_norm, norm_params):
    return action_norm * norm_params["action_std"] + norm_params["action_mean"]


def patch_pi0_cache_clone(policy):
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    model = policy.model

    def clone_cache(cache):
        if hasattr(cache, "__class__") and cache.__class__.__name__ == "DynamicCache":
            return cache.__class__.from_legacy_cache(cache.to_legacy_cache())
        if isinstance(cache, (tuple, list)):
            cloned = []
            for layer in cache:
                if isinstance(layer, (tuple, list)):
                    cloned.append(tuple(t.clone() if torch.is_tensor(t) else t for t in layer))
                else:
                    cloned.append(layer.clone() if torch.is_tensor(layer) else layer)
            return tuple(cloned) if isinstance(cache, tuple) else cloned
        return cache

    @torch.no_grad()
    def sample_actions_cache_clone(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=None, **kwargs):
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = state.shape[0]
        device = state.device
        if noise is None:
            noise = self.sample_noise((batch_size, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, prefix_cache = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time_value = 1.0 + step * dt
            time_tensor = torch.full((batch_size,), time_value, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=clone_cache(prefix_cache),
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    model.sample_actions = MethodType(sample_actions_cache_clone, model)


def tokenize_task(tokenizer, task_prompt, max_length):
    tokenized = tokenizer(
        task_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return (
        tokenized["input_ids"].to(DEVICE),
        tokenized["attention_mask"].bool().to(DEVICE),
    )


def scan_serial_ports():
    candidates = []
    for prefix in ["/dev/ttyACM", "/dev/ttyUSB", "/dev/tty.usbserial"]:
        for i in range(20):
            path = f"{prefix}{i}"
            if os.path.exists(path):
                candidates.append(path)
    return candidates


def scan_cameras(max_scan=10):
    available = []
    for i in range(max_scan):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        if cap is not None:
            cap.release()
    return available


class ArmController:
    def __init__(self, arm_port, camera_index):
        self.robot = None
        self.camera = None
        self.arm_port = arm_port
        self.camera_index = camera_index
        self._serial_lock = threading.Lock()
        self._cached_state = np.zeros(6, dtype=np.float32)
        self._connect_arm()
        self._connect_camera()

    def _connect_arm(self):
        print("  🦾 正在连接机械臂...")
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        robot_config = SOFollowerRobotConfig(port=self.arm_port)
        self.robot = SOFollower(robot_config)
        self.robot.connect(calibrate=False)
        print(f"  ✅ 机械臂已连接到 {self.arm_port}")

    def _connect_camera(self):
        print("  📷 正在连接摄像头...")
        connected = False
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            self.camera = cv2.VideoCapture(self.camera_index, backend)
            if self.camera.isOpened():
                connected = True
                break
            self.camera.release()

        if not connected:
            raise RuntimeError(f"无法打开摄像头 index={self.camera_index}")

        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, 156)
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 130)
        self.camera.set(cv2.CAP_PROP_CONTRAST, 20)
        self.camera.set(cv2.CAP_PROP_SATURATION, 80)
        self.camera.set(cv2.CAP_PROP_SHARPNESS, 3)
        self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.camera.set(cv2.CAP_PROP_GAIN, 0)

        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise RuntimeError("无法读取摄像头帧")
        print(f"  ✅ 摄像头已连接 (index={self.camera_index}): {actual_w}x{actual_h} @ {int(self.camera.get(cv2.CAP_PROP_FPS))}fps")

    def get_state(self):
        with self._serial_lock:
            obs = self.robot.get_observation()
            state = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
            self._cached_state = state.copy()
        return state

    def get_image(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise RuntimeError("读取摄像头帧失败")
        return frame

    def send_positions(self, target_dict):
        with self._serial_lock:
            return self.robot.send_action({f"{name}.pos": float(val) for name, val in target_dict.items()})

    def move_smooth(self, target_dict, duration=1.5):
        steps = max(1, int(duration * 30))
        current = self.get_state()
        targets = np.array([target_dict.get(n, current[i]) for i, n in enumerate(JOINT_NAMES)], dtype=np.float32)
        for step in range(steps):
            progress = (step + 1) / steps
            interp = current + (targets - current) * progress
            self.send_positions({n: interp[i] for i, n in enumerate(JOINT_NAMES)})
            time.sleep(1 / 30)

    def disconnect(self):
        if self.robot is not None:
            try:
                self.robot.disconnect()
                print("  机械臂已断开")
            except Exception:
                pass
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass


class Pi0VLAExecutor:
    def __init__(self, policy, tokenizer, norm_params, tokenizer_max_length, arm, show_window=True):
        self.policy = policy
        self.tokenizer = tokenizer
        self.norm = norm_params
        self.tokenizer_max_length = tokenizer_max_length
        self.arm = arm
        self.show_window = show_window
        self.last_action = np.zeros(6, dtype=np.float32)
        self.stop_event = threading.Event()

    def run_vla(self, task_prompt, max_steps=VLA_MAX_STEPS):
        print(f"  🚀 Pi0 VLA 任务: '{task_prompt}'")
        print(f"     稳定阈值: ±{VLA_STABLE_THRESHOLD} units × {VLA_STABLE_STEPS}步 | 最大步数: {max_steps}")
        print("     注意: 当前 SOFollower 默认使用归一化关节单位，不是角度")
        self.stop_event.clear()
        self.policy.reset()
        token_ids, attn_mask = tokenize_task(self.tokenizer, task_prompt, self.tokenizer_max_length)

        self.last_action = self.arm.get_state().copy()
        stable_count = 0
        result = "max_steps"
        step = 0

        try:
            while step < max_steps:
                if self.stop_event.is_set():
                    result = "interrupted"
                    break

                state = self.arm.get_state()
                image = self.arm.get_image()
                image_tensor = pi0_image_to_tensor(image)
                state_norm = normalize_state(state, self.norm)
                state_tensor = torch.from_numpy(state_norm).unsqueeze(0).float().to(DEVICE)
                batch = {
                    OBS_STATE: state_tensor,
                    "observation.images.front": image_tensor,
                    OBS_LANGUAGE_TOKENS: token_ids,
                    OBS_LANGUAGE_ATTENTION_MASK: attn_mask,
                }

                with torch.no_grad():
                    action_chunk_norm = self.policy.predict_action_chunk(batch)[0].cpu().numpy()
                action_chunk = unnormalize_action(action_chunk_norm, self.norm)

                for chunk_i, target in enumerate(action_chunk):
                    if step >= max_steps or self.stop_event.is_set():
                        result = "interrupted" if self.stop_event.is_set() else result
                        break

                    target = np.asarray(target, dtype=np.float32)
                    action_delta = np.abs(target - self.last_action)
                    max_delta = float(np.max(action_delta))
                    sent = self.arm.send_positions({name: target[i] for i, name in enumerate(JOINT_NAMES)})
                    self.last_action = target.copy()
                    time.sleep(1 / 30)
                    feedback = self.arm.get_state()
                    feedback_error = float(np.max(np.abs(feedback - target)))

                    if self.show_window:
                        self._visualize(image, feedback, target, step, max_delta, stable_count)

                    if max_delta < VLA_STABLE_THRESHOLD and feedback_error < 3.0:
                        stable_count += 1
                    else:
                        stable_count = 0

                    if step % 5 == 0 or stable_count == VLA_STABLE_STEPS:
                        status = f"稳定{stable_count}/{VLA_STABLE_STEPS}" if stable_count > 0 else "执行中"
                        sent_vals = [sent.get(f"{name}.pos", target[i]) for i, name in enumerate(JOINT_NAMES)] if sent else target
                        print(
                            f"    [步骤 {step:3d}.{chunk_i:02d}] Δcmd={max_delta:.2f} fb_err={feedback_error:.2f} | "
                            f"目标 pan={target[0]:+.1f} lift={target[1]:+.1f} elbow={target[2]:+.1f} grip={target[5]:+.1f} | "
                            f"发送 pan={sent_vals[0]:+.1f} lift={sent_vals[1]:+.1f} elbow={sent_vals[2]:+.1f} grip={sent_vals[5]:+.1f} | "
                            f"反馈 pan={feedback[0]:+.1f} lift={feedback[1]:+.1f} elbow={feedback[2]:+.1f} grip={feedback[5]:+.1f} | {status}"
                        )

                    if stable_count >= VLA_STABLE_STEPS:
                        print("\n    ✅ Pi0 输出和机械臂反馈已稳定，任务停止")
                        result = "stable"
                        break

                    step += 1

                self.policy.reset()
        finally:
            if self.show_window:
                try:
                    cv2.destroyWindow("Pi0 VLA Vision")
                except Exception:
                    pass

        if result == "stable":
            print(f"  ✅ Pi0 握手任务结束 (步骤 {step + 1})")
        elif result == "interrupted":
            print(f"  ⏹️  Pi0 任务被用户中断 (步骤 {step})")
        else:
            print(f"  ⚠️  Pi0 达到最大步数 ({max_steps})")
        return result

    def _visualize(self, image, state, action, step, delta, stable_count):
        vis = image.copy()
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(vis, f"Pi0 STEP {step:03d}  d={delta:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        bar_w = int((stable_count / VLA_STABLE_STEPS) * 200)
        cv2.rectangle(vis, (200, 15), (400, 35), (50, 50, 50), -1)
        cv2.rectangle(vis, (200, 15), (200 + bar_w, 35), (0, 150, 255), -1)
        cv2.putText(vis, f"STABLE:{stable_count}/{VLA_STABLE_STEPS}", (405, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, "Pi0 VLA - Handshake", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        bar_h = 130
        cv2.rectangle(vis, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        y = h - bar_h + 18
        for i, (name, s, a) in enumerate(zip(JOINT_NAMES, state, action)):
            label = f"{name[:12]}: s={s:+6.1f} -> a={a:+6.1f}"
            color = (0, 255, 255) if i < 3 else (0, 255, 0) if i < 5 else (255, 100, 100)
            cv2.putText(vis, label, (10, y + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow("Pi0 VLA Vision", vis)
        cv2.waitKey(1)


def wave_arm(arm):
    print("  👋 执行挥手...")
    ready_pos = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -20.0,
        "elbow_flex": 80.0,
        "wrist_flex": 60.0,
        "wrist_roll": 0.0,
        "gripper": 60.0,
    }
    arm.move_smooth(ready_pos, duration=1.2)
    time.sleep(0.3)
    for pan_angle in [-35, -20, 0, 20, 35, 20, 0, -20, -35, -20, 0, 20, 35, 20, 0, -20]:
        cur = arm.get_state()
        target = {name: cur[i] for i, name in enumerate(JOINT_NAMES)}
        target["shoulder_pan"] = float(pan_angle)
        arm.move_smooth(target, duration=0.15)
        time.sleep(0.05)
    arm.move_smooth(ready_pos, duration=0.5)
    print("  ✅ 挥手完成")


def open_gripper(arm):
    print("  ✋ 打开夹爪...")
    cur = arm.get_state()
    target = {name: cur[i] for i, name in enumerate(JOINT_NAMES)}
    target["gripper"] = 80.0
    arm.move_smooth(target, duration=1.0)
    print("  ✅ 夹爪已打开")


def close_gripper(arm):
    print("  ✊ 关闭夹爪...")
    cur = arm.get_state()
    target = {name: cur[i] for i, name in enumerate(JOINT_NAMES)}
    target["gripper"] = 0.0
    arm.move_smooth(target, duration=1.0)
    print("  ✅ 夹爪已关闭")


def go_to_zero(arm):
    print("  🏠 回到零位...")
    arm.move_smooth({name: 0.0 for name in JOINT_NAMES}, duration=2.0)
    print("  ✅ 已回到零位")


def execute_action_command(arm, user_input):
    text = user_input.lower().strip()
    current_state = arm.get_state()
    joint_map = {
        "大臂": ("shoulder_lift", 1, 30),
        "小臂": ("elbow_flex", 2, 25),
        "手腕": ("wrist_flex", 3, 20),
        "手腕旋转": ("wrist_roll", 4, 25),
        "肩部转动": ("shoulder_pan", 0, 20),
        "夹爪": ("gripper", 5, 15),
    }
    direction_map = {
        "抬高": -1, "向上": -1, "升起": -1, "抬起": -1,
        "降低": 1, "向下": 1, "降下": 1, "放下": 1,
        "向左": -1, "左转": -1, "左旋": -1,
        "向右": 1, "右转": 1, "右旋": 1,
        "打开": 1, "张开": 1, "松开": 1,
        "关闭": -1, "闭合": -1, "夹紧": -1,
    }

    if any(k in text for k in ["摄像头", "视角", "视野", "画面"]):
        if any(k in text for k in ["抬高", "向上", "抬"]):
            new_state = current_state.copy()
            new_state[2] -= 15
            new_state[3] += 15
            arm.move_smooth({JOINT_NAMES[i]: new_state[i] for i in range(6)}, duration=1.0)
            return True, "已抬高摄像头视角"
        if any(k in text for k in ["降低", "向下", "降"]):
            new_state = current_state.copy()
            new_state[2] += 15
            new_state[3] -= 15
            arm.move_smooth({JOINT_NAMES[i]: new_state[i] for i in range(6)}, duration=1.0)
            return True, "已降低摄像头视角"

    for cn_name, (en_name, idx, max_delta) in joint_map.items():
        if cn_name in text or en_name.replace("_", "") in text:
            for direction, sign in direction_map.items():
                if direction in text:
                    new_angle = current_state[idx] + max_delta * sign
                    if idx == 5:
                        new_angle = max(0, min(80, new_angle))
                    arm.move_smooth({en_name: new_angle}, duration=1.0)
                    return True, f"已{direction}{cn_name}: {current_state[idx]:+.1f}° → {new_angle:+.1f}°"
    return False, None


def parse_action_command(user_input):
    text = user_input.lower().strip()
    action_keywords = [
        "抬高", "降低", "向上", "向下", "升起", "降下", "抬起", "放下",
        "向左", "向右", "左转", "右转", "左旋", "右旋",
        "打开", "关闭", "张开", "闭合", "松开", "夹紧",
        "摄像头", "视角", "视野", "画面",
        "大臂", "小臂", "手腕", "肩部", "夹爪",
        "移动", "转动", "旋转", "伸展", "弯曲",
    ]
    existing_cmds = ["挥手", "wave", "零位", "回零", "复位", "reset", "看", "观察", "退出", "exit", "帮助", "help", "握手", "handshake"]
    if any(k in text for k in action_keywords) and not any(k in text for k in existing_cmds):
        return "action"
    return None


def _analyze_scene(image, yolo_model=None):
    h, w = image.shape[:2]
    descriptions = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:
        descriptions.append("环境偏暗")
    elif brightness > 200:
        descriptions.append("环境很亮")

    detected_objects = []
    if yolo_model is not None:
        try:
            results = yolo_model(image, verbose=False)
            if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results[0].names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    pos_x = "左" if cx < w * 0.35 else "右" if cx > w * 0.65 else "中"
                    pos_y = "上" if cy < h * 0.4 else "下" if cy > h * 0.7 else "中"
                    rel_size = ((x2 - x1) * (y2 - y1)) / (w * h) * 100
                    size_desc = "大" if rel_size > 15 else "中" if rel_size > 5 else "小"
                    detected_objects.append({"label": label, "conf": conf, "pos": f"{pos_x}{pos_y}", "size": size_desc})
        except Exception as e:
            descriptions.append(f"[YOLO检测异常: {e}]")

    if detected_objects:
        detected_objects.sort(key=lambda o: o["conf"], reverse=True)
        descriptions.append("看到: " + ", ".join(f"{obj['label']}({obj['pos']}{obj['size']} {obj['conf']:.0%})" for obj in detected_objects[:8]))
    else:
        descriptions.append("未检测到已知物体")
    return "；".join(descriptions)


def init_llm_client():
    ark_key = os.environ.get("ARK_API_KEY", "")
    ark_base = os.environ.get("ARK_BASE_URL", "")
    ark_model = os.environ.get("ARK_MODEL_ID", "")
    if ark_key and "your_api" not in ark_key:
        print(f"  🤖 使用火山引擎(ARK) LLM: {ark_model}")
        return OpenAI(api_key=ark_key, base_url=ark_base), ark_model

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key and "your_api" not in ds_key:
        print("  🤖 使用 DeepSeek LLM")
        return OpenAI(api_key=ds_key, base_url="https://api.deepseek.com"), "deepseek-chat"

    oa_key = os.environ.get("OPENAI_API_KEY", "")
    if oa_key:
        print("  🤖 使用 OpenAI LLM")
        return OpenAI(api_key=oa_key), "gpt-4o-mini"
    return None, None


def describe_scene_with_llm(llm_client, llm_model, scene_desc, arm_state):
    if llm_client is None or llm_model is None:
        return f"[LLM未配置] {scene_desc}"

    state_text = (
        f"肩_pan={arm_state[0]:+.1f}° 肩_lift={arm_state[1]:+.1f}° "
        f"肘={arm_state[2]:+.1f}° 腕_flex={arm_state[3]:+.1f}° "
        f"腕_roll={arm_state[4]:+.1f}° 夹爪={arm_state[5]:+.1f}°"
    )
    prompt = (
        "你是一个机械臂视觉助手。根据以下摄像头检测结果，用简洁自然的中文描述你看到的内容。\n"
        "要求：直接描述看到的物体和位置，不要编造，2-3句话。\n\n"
        f"检测结果: {scene_desc}\n"
        f"机械臂关节角度: {state_text}\n\n"
        "请直接输出描述："
    )
    try:
        resp = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM调用失败: {e}] {scene_desc}"


def observe(arm, yolo_model=None, llm_client=None, llm_model=None):
    print("  👁 正在观察...")
    try:
        image = arm.get_image()
        state = arm.get_state()
        scene_desc = _analyze_scene(image, yolo_model=yolo_model)
        print()
        print(f"  📷 [视觉分析] {scene_desc}")
        print(
            f"  🦾 [机械臂状态] 肩_pan={state[0]:+.1f}° 肩_lift={state[1]:+.1f}° "
            f"肘={state[2]:+.1f}° 腕_flex={state[3]:+.1f}° 腕_roll={state[4]:+.1f}° 夹爪={state[5]:+.1f}°"
        )
        print("  🤖 正在生成场景描述...", end="", flush=True)
        llm_desc = describe_scene_with_llm(llm_client, llm_model, scene_desc, state)
        print()
        print(f"  💬 [场景描述] {llm_desc}")
        print()

        vis = image.copy()
        if yolo_model is not None:
            try:
                yolo_results = yolo_model(image, verbose=False)
                if yolo_results and len(yolo_results) > 0 and hasattr(yolo_results[0], "boxes") and yolo_results[0].boxes is not None:
                    vis = yolo_results[0].plot()
            except Exception:
                pass

        _, vw = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (vw, 160), (0, 0, 0), -1)
        cv2.putText(vis, "OBSERVE MODE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        desc_text = llm_desc if llm_desc else scene_desc
        cv2.putText(vis, desc_text[:55], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if len(desc_text) > 55:
            cv2.putText(vis, desc_text[55:110], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        y = 95
        for i, (name, s) in enumerate(zip(JOINT_NAMES, state)):
            color = (0, 255, 255) if i < 3 else (0, 255, 0) if i < 5 else (255, 100, 100)
            cv2.putText(vis, f"{name}: {s:+7.2f}°", (10, y + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.imshow("Observe - Press any key to exit", vis)
        for _ in range(60):
            key = cv2.waitKey(50) & 0xFF
            if key != 255:
                break
        cv2.destroyWindow("Observe - Press any key to exit")
        print("  ✅ 观察结束")
    except Exception as e:
        print(f"  ⚠️  观察失败: {e}")


def parse_command(user_input):
    text = user_input.lower().strip()
    if any(k in text for k in ["握手", "handshake", "握手任务", "shake", "握个手"]):
        return "handshake"
    if any(k in text for k in ["挥手", "wave", "挥挥手", "打招呼"]):
        return "wave"
    if any(k in text for k in ["打开夹爪", "open", "张开夹爪"]):
        return "open_gripper"
    if any(k in text for k in ["关闭夹爪", "close", "闭合夹爪", "夹紧夹爪"]):
        return "close_gripper"
    if any(k in text for k in ["零位", "回零", "复位", "reset", "home"]):
        return "zero"
    if any(k in text for k in ["看", "show", "看看", "观察", "看一下"]):
        return "observe"
    if any(k in text for k in ["退出", "exit", "quit", "再见", "拜拜"]):
        return "exit"
    if any(k in text for k in ["帮助", "help", "?", "？", "怎么用"]):
        return "help"
    action_cmd = parse_action_command(user_input)
    if action_cmd:
        return action_cmd
    return "chat"


def print_help(pi0_available=False):
    print()
    print("=" * 55)
    print("  帮助信息:")
    print("=" * 55)
    print("  内置动作:")
    print("    - 挥手 / wave        -> 机械臂挥手打招呼")
    print("    - 打开夹爪 / open    -> 打开夹爪")
    print("    - 关闭夹爪 / close   -> 关闭夹爪")
    print("    - 回到零位 / reset   -> 机械臂回到零位")
    print()
    print("  Pi0 VLA 任务:")
    print("    - 握手 / handshake   -> Pi0 VLA 握手任务" + (" [已加载]" if pi0_available else " [模型未加载]"))
    print()
    print("  自然语言动作命令:")
    print("    - 抬高/降低 摄像头、大臂、小臂、手腕")
    print("    - 向左/向右 转动")
    print("    - 打开/关闭 夹爪")
    print()
    print("  其他:")
    print("    - 观察 / observe     -> 显示摄像头+关节状态+LLM描述")
    print("    - 退出 / exit        -> 退出程序")
    print("=" * 55)


def generate_chat_reply(user_input, llm_client=None, llm_model=None):
    if llm_client is not None and llm_model is not None:
        try:
            resp = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "你是朝野小助手，一个能控制机械臂的AI助手。回答简洁自然，中文，1-2句话。"},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.7,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    text = user_input.lower()
    if any(k in text for k in ["你好", "hi", "hello", "您好", "嗨"]):
        return "你好呀！我可以帮你控制机械臂挥手、观察或执行 Pi0 握手任务。"
    if any(k in text for k in ["谢谢", "thank", "thanks"]):
        return "不客气！有什么我可以帮你的吗？"
    if any(k in text for k in ["你是谁", "who", "名字"]):
        return "我是朝野小助手，一个能控制机械臂的助手。"
    if any(k in text for k in ["时间", "time"]):
        return f"现在是 {time.strftime('%Y-%m-%d %H:%M:%S')}。"
    return "嗯，我不太确定你想让我做什么。你可以让我'挥手'、'握手'、'观察'或者输入'帮助'。"


def choose_serial_port():
    print("🔍 扫描串口设备...")
    available_ports = scan_serial_ports()
    if available_ports:
        print()
        print("  可用的串口设备:")
        for i, port in enumerate(available_ports):
            print(f"    [{i + 1}] {port}")
        print()
        while True:
            try:
                choice = input("  🦾 选择机械臂端口 (输入编号, 默认1): ").strip() or "1"
                idx = int(choice) - 1
                if 0 <= idx < len(available_ports):
                    return available_ports[idx]
                print(f"  ⚠️  请输入 1-{len(available_ports)} 之间的数字")
            except ValueError:
                print("  ⚠️  请输入有效的数字")
    print()
    print("  ⚠️  没有自动检测到串口设备，请手动输入端口路径")
    return input("  🦾 输入机械臂端口 (例如 /dev/ttyACM0): ").strip()


def choose_camera():
    print("🔍 扫描摄像头设备...")
    available_cams = scan_cameras()
    if available_cams:
        print()
        print("  可用的摄像头:")
        for i, idx in enumerate(available_cams):
            print(f"    [{i + 1}] index={idx}")
        print()
        while True:
            try:
                choice = input("  📷 选择摄像头 (输入编号, 默认1): ").strip() or "1"
                idx = int(choice) - 1
                if 0 <= idx < len(available_cams):
                    return available_cams[idx]
                print(f"  ⚠️  请输入 1-{len(available_cams)} 之间的数字")
            except ValueError:
                print("  ⚠️  请输入有效的数字")
    print()
    print("  ⚠️  没有自动检测到摄像头，请手动输入摄像头索引")
    try:
        return int(input("  📷 输入摄像头索引 (例如 0): ").strip())
    except ValueError:
        print("  ❌ 无效输入，默认使用 0")
        return 0


def main():
    print("=" * 60)
    print("朝野小助手 - Pi0 VLA 机械臂控制台")
    print("=" * 60)
    print()
    print("正在初始化系统...")
    print()

    arm_port = choose_serial_port()
    print(f"  → 已选择: {arm_port}")
    print()
    camera_index = choose_camera()
    print(f"  → 已选择: index={camera_index}")
    print()

    arm = ArmController(arm_port=arm_port, camera_index=camera_index)
    print()

    pi0_executor = None
    pi0_available = False
    try:
        pi0_policy, pi0_tokenizer, pi0_norm, tokenizer_max_length = load_pi0_policy()
        pi0_executor = Pi0VLAExecutor(pi0_policy, pi0_tokenizer, pi0_norm, tokenizer_max_length, arm, show_window=True)
        pi0_available = True
        print("  ✅ Pi0 VLA 已加载 (握手任务就绪)")
        print()
    except Exception as e:
        print(f"  ⚠️ Pi0 VLA 模型加载失败: {e}")
        print("     暂时无法执行握手任务，但其他功能正常可用")
        print()

    yolo_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11x.pt")
    yolo_model = None
    if os.path.exists(yolo_model_path):
        try:
            yolo_model = YOLO(yolo_model_path)
            print(f"  ✅ YOLO 目标检测已加载 ({yolo_model_path})")
        except Exception as e:
            print(f"  ⚠️ YOLO 加载失败: {e}")
            print("     观察功能将使用基础模式")
    else:
        print(f"  ⚠️ 未找到 YOLO 模型: {yolo_model_path}")
        print("     观察功能将使用基础模式")

    llm_client, llm_model_name = init_llm_client()
    if llm_client is None:
        print("  ⚠️ 未配置 LLM API Key，观察描述将使用原始检测结果")
        print("     可在 .env 文件中配置 ARK_API_KEY 或 DEEPSEEK_API_KEY")

    print("=" * 60)
    print()
    time.sleep(0.3)
    print("🤖 你好，我是朝野的小助手，请问我能为你做什么？")
    print()
    print("可用命令:")
    print("  🤝  握手 / handshake      - Pi0 VLA 握手任务")
    print("  👋  挥手 / wave           - 机械臂挥手")
    print("  ✋  打开夹爪 / open        - 打开夹爪")
    print("  ✊  关闭夹爪 / close       - 关闭夹爪")
    print("  🏠  回到零位 / reset       - 回到零位")
    print("  👁️  观察 / observe         - 显示摄像头+关节状态")
    print("  ❓  帮助 / help            - 显示帮助")
    print("  👋  退出 / exit            - 退出程序")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("👤 你: ").strip()
            if not user_input:
                continue

            cmd = parse_command(user_input)
            print()

            if cmd == "handshake":
                if not pi0_available:
                    print("🤖 抱歉，Pi0 VLA 模型未加载成功，暂时无法执行握手任务")
                    print("   请检查 Pi0 模型路径和依赖后重试")
                else:
                    print("🤖 好的，启动 Pi0 VLA 握手任务...")
                    try:
                        result = pi0_executor.run_vla("shake hands", max_steps=VLA_MAX_STEPS)
                        if result == "stable":
                            print("🤖 握手任务结束！")
                        else:
                            print("🤖 握手任务结束（可能未完全完成）")
                    except Exception as e:
                        print(f"  ⚠️ 握手异常: {e}")
            elif cmd == "wave":
                print("🤖 好的，我来挥手向你问好！")
                wave_arm(arm)
            elif cmd == "open_gripper":
                print("🤖 好的，正在打开夹爪...")
                open_gripper(arm)
            elif cmd == "close_gripper":
                print("🤖 好的，正在关闭夹爪...")
                close_gripper(arm)
            elif cmd == "zero":
                print("🤖 好的，正在回到零位...")
                go_to_zero(arm)
            elif cmd == "observe":
                print("🤖 好的，让我看看周围环境...")
                observe(arm, yolo_model=yolo_model, llm_client=llm_client, llm_model=llm_model_name)
            elif cmd == "action":
                success, msg = execute_action_command(arm, user_input)
                if success:
                    print(f"🤖 {msg}")
                else:
                    print("🤖 抱歉，无法识别这个动作命令")
                    print("   支持的格式: 抬高/降低 + 摄像头/大臂/小臂/手腕")
            elif cmd == "help":
                print_help(pi0_available)
            elif cmd == "exit":
                print("🤖 再见！感谢使用朝野小助手 👋")
                break
            elif cmd == "chat":
                reply = generate_chat_reply(user_input, llm_client=llm_client, llm_model=llm_model_name)
                print(f"🤖 {reply}")
            print()

        except KeyboardInterrupt:
            print("\n\n🤖 检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n正在关闭系统...")
    arm.disconnect()
    cv2.destroyAllWindows()
    print("✅ 程序已退出")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被中断")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    except Exception as e:
        print(f"\n\n致命错误: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
