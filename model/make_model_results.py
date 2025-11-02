import cv2
import os
from ultralytics import YOLO

# 1. 모델 불러오기
model = YOLO('./runs/detect/Soldier_Detection3/weights/best.pt')

# 2. 입력 폴더 및 출력 폴더 지정
input_dir = './data/test/camouflage'
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 3. 지원할 이미지 확장자
exts = ('.jpg', '.jpeg', '.png')

# 4. 폴더 내 모든 이미지 순회
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(exts):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # 5. 모델 추론
    results = model.predict(img, verbose=False)

    # 6. 박스 시각화 (confidence ≥ 0.5)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 박스 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # 라벨 (confidence)
                # label = f"{conf:.2f}"
                # cv2.putText(img, label, (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 7. 결과 이미지 저장
    save_name = os.path.splitext(filename)[0] + '.png'
    save_path = os.path.join(output_dir, f"result_{filename}")
    cv2.imwrite(save_path, img)
    print(f"저장 완료: {save_path}")

print("✅ 모든 이미지 추론 및 저장 완료!")
