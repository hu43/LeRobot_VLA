# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_vr")
@dataclass
class XLerobotVRTeleopConfig(TeleoperatorConfig):
    """Config for the XLeRobot VR teleoperator (browser WebXR, e.g. Meta Quest)."""

    # Path to the XLeVR package (contains xlevr/, web-ui/, cert.pem, key.pem)
    xlevr_path: str = "/home/zhoulin/hu/lerobot/XLeRobot/XLeVR"
    # How long connect() waits for the VR browser client to send first data
    vr_connection_timeout: float = 60.0
    # Proportional gain for joint-space P-control (1.0 sends absolute targets)
    kp: float = 1.0
    # IP shown in the connection hint for the VR browser. Servers always bind
    # 0.0.0.0; set this when auto-detection picks the wrong interface (e.g. VPN).
    display_host: str | None = None
