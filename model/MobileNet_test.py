import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1) 경로 설정
model_path = 'mobilenetv3_epoch8_acc0.9977.pth'
# image_path = './data/test/images/000061.jpg'  # 테스트할 이미지 경로
image_path = './test_image/KakaoTalk_20250529_200322276.jpg'  # 테스트할 이미지 경로

# 2) 클래스 라벨 정의 (ImageFolder 기준)
class_names = ['camouflage', 'noncamouflage']  # ImageFolder의 폴더 이름 순서에 맞춰야 함

# 3) 모델 준비
model = models.mobilenet_v3_small(pretrained=False)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, 2)  # 이진 분류
model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 4) 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 5) 이미지 로드 및 추론
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
output = model(input_tensor)
pred = torch.argmax(output, dim=1).item()
confidence = torch.softmax(output, dim=1)[0][pred].item()

# 6) 결과 출력
plt.imshow(image)
plt.axis('off')
plt.title(f'Prediction: {class_names[pred]} ({confidence:.2%})')
plt.show()