from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 1. 경로 설정
model_path = "./runs/detect/Soldier_Detection5/weights/best.pt"   # 학습된 YOLOv8 모델
data_dir = "./data/test/images"                   # 테스트 이미지 폴더
label_dir = "./data/test/labels"                  # YOLO 라벨 txt 폴더
class_names = ['camouflage', 'noncamouflage']        # 클래스 순서

# 2. 모델 로드
model = YOLO(model_path)
model.conf = 0.25  # confidence threshold

# 3. 결과 수집
all_preds = []
all_trues = []

# 4. 이미지 순회
for fname in tqdm(sorted(os.listdir(data_dir))):
    if not fname.endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(data_dir, fname)
    label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt'))

    # 4-1. GT 라벨 불러오기
    true_classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls_id = int(line.strip().split()[0])
                true_classes.append(cls_id)

    # 4-2. 예측 수행
    results = model.predict(source=image_path, save=False, conf=0.25, iou=0.45, verbose=False)[0]
    pred_classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []

    # 4-3. 각 클래스별 존재 여부 판단 (멀티라벨 → 단일 예측으로 전환)
    # GT: 이 이미지에 camouflage(0) 있다면 0 포함, 없다면 1로만 구성
    if 0 in true_classes:
        all_trues.append(0)
    else:
        all_trues.append(1)

    if 0 in pred_classes:
        all_preds.append(0)
    else:
        all_preds.append(1)

# 5. 평가
cm = confusion_matrix(all_trues, all_preds)
print("\n[Confusion Matrix]")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("YOLOv8 Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n[Classification Report]")
print(classification_report(all_trues, all_preds, target_names=class_names))