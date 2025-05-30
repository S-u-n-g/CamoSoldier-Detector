import time
import numpy as np
from ultralytics import YOLO
import cv2
import torch

# 1. YOLO 모델 로드
model = YOLO("runs/detect/Soldier_Detection3/weights/best.pt")  # 모델 경로 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 테스트용 이미지 (더미 또는 실제)
use_webcam = False
if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# 3. 반복 설정
num_trials = 10
num_frames = 100
fps_list = []

for trial in range(num_trials):
    start_time = time.time()

    for _ in range(num_frames):
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            img = frame
        else:
            img = dummy_image.copy()

        _ = model.predict(img, imgsz=224, verbose=False)

    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    fps_list.append(fps)
    print(f"[{trial+1}회차] FPS: {fps:.2f} frames/sec")

if use_webcam:
    cap.release()

# 4. 평균 FPS 출력
avg_fps = sum(fps_list) / len(fps_list)
print(f"\n[🔥] 평균 FPS (10회 측정 기준): {avg_fps:.2f} frames/sec")
