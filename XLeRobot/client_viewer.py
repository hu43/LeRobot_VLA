import zmq
import json
import base64
import numpy as np
import cv2

def run_client(remote_ip="10.8.0.16"):
    context = zmq.Context()
    # 对应服务器的 PUSH，本地需要 PULL
    data_socket = context.socket(zmq.PULL)
    data_socket.connect(f"tcp://{remote_ip}:5556")
    
    print(f"正在连接到 {remote_ip} 接收画面...")

    while True:
        try:
            # 接收 JSON 数据
            message = data_socket.recv_string()
            data = json.loads(message)
            
            if data.get("response") == "video":
                # 解码 Base64 图像数据
                img_b64 = data["data"]["frame"]
                img_data = base64.b64decode(img_b64)
                
                # 转换回 OpenCV 格式并显示
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                cv2.imshow("XLeRobot Remote View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"接收出错: {e}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_client()