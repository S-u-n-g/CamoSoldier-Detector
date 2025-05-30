from ultralytics import YOLO

# 모델 로드
model = YOLO('./runs/detect/Soldier_Detection3/weights/best.pt')

# 평가 수행
metrics = model.val(data='./data/data.yaml')

# 정확도 지표 출력
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Mean Precision: {metrics.box.mp:.4f}")
print(f"Mean Recall: {metrics.box.mr:.4f}")

# 추론 시간 정보 출력
print(f"Inference time per image: {metrics.speed.get('inference', 0):.2f} ms")
print(f"NMS time per image: {metrics.speed.get('nms', 0):.2f} ms")
print(f"Preprocess time per image: {metrics.speed.get('preprocess', 0):.2f} ms")
print(f"Postprocess time per image: {metrics.speed.get('postprocess', 0):.2f} ms")

# 전체 시간 합산
total = sum(metrics.speed.values())
print(f"Total time per image: {total:.2f} ms")

