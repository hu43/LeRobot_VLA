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

"""
VR Monitor for the XLeVR package.

Adapted from XLeRobot/XLeVR/vr_monitor.py: the hardcoded XLEVR_PATH and the
permanent os.chdir() were removed. The path to XLeVR is injected via the
constructor and SSL certificate paths are resolved to absolute paths so the
monitor can run from any working directory.
"""

import asyncio
import http.server
import logging
import os
import socket
import ssl
import sys
import threading

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "localhost"


class SimpleAPIHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP request handler serving the XLeVR web-ui."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_file("web-ui/index.html", "text/html")
        elif self.path.endswith(".css"):
            self.serve_file(f"web-ui{self.path}", "text/css")
        elif self.path.endswith(".js"):
            self.serve_file(f"web-ui{self.path}", "application/javascript")
        elif self.path.endswith(".ico"):
            self.serve_file(self.path[1:], "image/x-icon")
        elif self.path.endswith((".jpg", ".jpeg", ".png", ".gif")):
            content_type = (
                "image/jpeg"
                if self.path.endswith((".jpg", ".jpeg"))
                else "image/png"
                if self.path.endswith(".png")
                else "image/gif"
            )
            self.serve_file(f"web-ui{self.path}", content_type)
        else:
            self.send_error(404, "Not found")

    def serve_file(self, filename, content_type):
        try:
            web_root = getattr(self.server, "web_root_path", os.getcwd())
            file_path = os.path.join(web_root, filename)

            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    content = f.read()

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception:
            self.send_error(500, "Internal server error")


class SimpleHTTPSServer:
    """HTTPS server serving the web-ui to the VR headset browser."""

    def __init__(self, config, web_root: str):
        self.config = config
        self.httpd = None
        self.server_thread = None
        self.web_root_path = web_root

    async def start(self):
        self.httpd = http.server.HTTPServer((self.config.host_ip, self.config.https_port), SimpleAPIHandler)
        self.httpd.web_root_path = self.web_root_path

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.config.certfile, self.config.keyfile)
        self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)

        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        logger.info(f"HTTPS server started on {self.config.host_ip}:{self.config.https_port}")

    async def stop(self):
        if self.httpd:
            self.httpd.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=5)
        logger.info("HTTPS server stopped")


class VRMonitor:
    """Runs the XLeVR websocket server and web-ui HTTPS server in one loop."""

    def __init__(self, xlevr_path: str, display_host: str | None = None):
        self.xlevr_path = os.path.abspath(os.path.expanduser(xlevr_path))
        self.display_host = display_host
        self.config = None
        self.vr_server = None
        self.https_server = None
        self.is_running = False
        self.left_goal = None
        self.right_goal = None
        self.headset_goal = None
        self._goal_lock = threading.Lock()

    def _import_xlevr_modules(self):
        # xlevr/config.py loads config.yaml relative to the CWD at import
        # time, so we chdir into the XLeVR directory only for the import.
        if not os.path.isdir(self.xlevr_path):
            raise FileNotFoundError(f"XLeVR path does not exist: {self.xlevr_path}")

        prev_cwd = os.getcwd()
        os.chdir(self.xlevr_path)
        try:
            if self.xlevr_path not in sys.path:
                sys.path.insert(0, self.xlevr_path)
            from xlevr.config import XLeVRConfig
            from xlevr.inputs.base import ControlGoal, ControlMode  # noqa: F401
            from xlevr.inputs.vr_ws_server import VRWebSocketServer

            return XLeVRConfig, VRWebSocketServer, ControlGoal, ControlMode
        finally:
            os.chdir(prev_cwd)

    def initialize(self) -> bool:
        try:
            XLeVRConfig, VRWebSocketServer, ControlGoal, ControlMode = self._import_xlevr_modules()
        except ImportError as e:
            logger.error(
                f"Failed to import xlevr modules from {self.xlevr_path}: {e}. "
                "Install XLeVR dependencies (websockets, scipy): uv pip install websockets scipy"
            )
            return False

        self.config = XLeVRConfig()
        self.config.enable_vr = True
        self.config.enable_keyboard = False
        self.config.enable_https = True
        # Resolve certificates relative to the XLeVR directory, not the CWD.
        self.config.certfile = os.path.join(self.xlevr_path, "cert.pem")
        self.config.keyfile = os.path.join(self.xlevr_path, "key.pem")

        self.command_queue = asyncio.Queue()

        try:
            self.vr_server = VRWebSocketServer(
                command_queue=self.command_queue,
                config=self.config,
                print_only=False,
            )
        except Exception as e:
            logger.error(f"Failed to create VR WebSocket server: {e}")
            return False

        try:
            self.https_server = SimpleHTTPSServer(self.config, self.xlevr_path)
        except Exception as e:
            logger.error(f"Failed to create HTTPS server: {e}")
            return False

        return True

    async def start_monitoring(self):
        if not self.initialize():
            logger.error("Failed to initialize VR monitor")
            return

        try:
            await self.https_server.start()
            await self.vr_server.start()

            self.is_running = True
            host_display = (
                self.display_host
                or (get_local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip)
            )
            logger.info(
                "VR Monitor running. Open your VR headset browser and navigate to: "
                f"https://{host_display}:{self.config.https_port}"
            )

            await self.monitor_commands()
        except Exception as e:
            logger.error(f"Error in VR monitor: {e}")
        finally:
            await self.stop_monitoring()

    async def monitor_commands(self):
        while self.is_running:
            try:
                goal = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                with self._goal_lock:
                    if goal.arm == "left":
                        self.left_goal = goal
                    elif goal.arm == "right":
                        self.right_goal = goal
                    elif goal.arm == "headset":
                        self.headset_goal = goal
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing VR command: {e}")

    def get_latest_goal_nowait(self, arm=None):
        """Return the latest VR control goal, or a dict with both arms' goals."""
        with self._goal_lock:
            if arm == "left":
                return self.left_goal
            elif arm == "right":
                return self.right_goal
            elif arm == "headset":
                return self.headset_goal
            else:
                return {
                    "left": self.left_goal,
                    "right": self.right_goal,
                    "headset": self.headset_goal,
                    "has_left": self.left_goal is not None,
                    "has_right": self.right_goal is not None,
                    "has_headset": self.headset_goal is not None,
                }

    def has_vr_data(self) -> bool:
        with self._goal_lock:
            return self.left_goal is not None or self.right_goal is not None

    async def stop_monitoring(self):
        self.is_running = False
        if self.vr_server:
            try:
                await self.vr_server.stop()
            except Exception:
                pass
        if self.https_server:
            try:
                await self.https_server.stop()
            except Exception:
                pass
        logger.info("VR Monitor stopped")

    def stop(self):
        """Best-effort synchronous stop (websocket loop lives in its own thread)."""
        self.is_running = False
        if self.https_server and self.https_server.httpd:
            try:
                self.https_server.httpd.shutdown()
            except Exception:
                pass
