import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import time
import numpy as np

# 모델 로드
model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("mobilenetv3_epoch8_acc0.9977.pth", map_location='cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 더미 이미지
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 반복 설정
num_trials = 10
num_frames = 1000
fps_list = []

with torch.no_grad():
    for trial in range(num_trials):
        start_time = time.time()
        for i in range(num_frames):
            img = dummy_image.copy()
            input_tensor = transform(img).unsqueeze(0).to(device)
            output = model(input_tensor)
        end_time = time.time()

        total_time = end_time - start_time
        fps = num_frames / total_time
        fps_list.append(fps)
        print(f"[{trial+1}회차] FPS: {fps:.2f} frames/sec")

# 평균 FPS 출력
avg_fps = sum(fps_list) / len(fps_list)
print(f"\n[🔥] 평균 FPS (10회 측정 기준): {avg_fps:.2f} frames/sec")
