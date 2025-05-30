import cv2
from ultralytics import YOLO

if __name__ ==  '__main__':
    # Load a model
    model = YOLO("yolov8n.pt")

    # Train the model on your custom dataset
    train_result = model.train(
        data="C:/Users/anmin/PycharmProjects/Graduation Project/Camouflage_Soldier_Detection/data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="Soldier Detection"
    )

    # 모델 평가
    val_result = model.val()

    # Mean Results 확인
    mean_results = val_result.mean_results()
    print("Mean Results:", mean_results)


    img = cv2.imread("C:/Users/anmin/PycharmProjects/Graduation Project/Camouflage_Soldier_Detection/data/test/images/000061.jpg.jpg")

    # Perform object detection
    results = model.predict("C:/Users/anmin/PycharmProjects/Graduation Project/Camouflage_Soldier_Detection/data/test/images/1.jpg")

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

        # 디버깅: 감지된 박스가 있는지 확인
        if len(boxes) == 0:
            print("Warning: No boxes detected.")
        else:
            print(f"{len(boxes)} boxes detected.")

        # Loop over detected boxes and draw them on the image
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{cls} {conf:.2f}"

            # Put class label and confidence score on the image
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save the result image with bounding boxes
    cv2.imwrite('result_with_boxes.jpg', img)
    # Display the image with bounding boxes
    cv2.namedWindow('Image with Bounding Boxes', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
