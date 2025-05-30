import cv2
from ultralytics import YOLO
import time

cap = cv2.VideoCapture(0)
model = YOLO('./runs/detect/Soldier_Detection3/weights/best.pt')
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    result = results[0]
    boxes = result.boxes

    # FPS 계산
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # FPS 표시
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            if cls_id == 0:
                # 블러 처리 없이 박스만 그림
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # 클래스와 신뢰도도 같이 표시
                label = f"soldier {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Soldier Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()