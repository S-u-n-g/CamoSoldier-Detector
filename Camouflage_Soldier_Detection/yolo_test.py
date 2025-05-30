import cv2
from ultralytics import YOLO

# 1. 모델 불러오기
model = YOLO('./runs/detect/Soldier_Detection3/weights/best.pt')

# 2. 이미지 불러오기
img_path = 'C:/Users/anmin/PycharmProjects/Graduation Project/Camouflage_Soldier_Detection/data/test/images/000060.jpg'
img = cv2.imread(img_path)

# 3. 모델로 예측 (predict)
results = model.predict(img)

# 4. 박스 시각화 (confidence ≥ 0.5)
for result in results:
    boxes = result.boxes
    for box in boxes:
        conf = float(box.conf[0])
        if conf >= 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # 라벨 텍스트
            label = f"{conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 결과 출력
cv2.imshow('result_with_boxes', img)
cv2.imwrite('result_with_boxes.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()