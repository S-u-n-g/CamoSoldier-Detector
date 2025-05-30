import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mobilenetv3_epoch8_acc0.9977.pth"
test_dir = "./dataset/test"  # 폴더 구조: test/camouflage, test/noncamouflageq

# 2. 전처리 (학습 시와 동일해야 함)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 데이터셋 & DataLoader
test_ds = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
class_names = test_ds.classes  # ['camouflage', 'noncamouflage']

# 4. 모델 불러오기
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 5. 오분류 이미지 저장용 리스트
wrong_images = []
wrong_preds = []
wrong_labels = []

# 6. 테스트셋 전체 순회하며 오분류 탐지
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        labels = labels.cpu()

        for i in range(len(images)):
            if preds[i] != labels[i]:
                wrong_images.append(images[i].cpu())
                wrong_preds.append(preds[i].item())
                wrong_labels.append(labels[i].item())

print(f"❗ 오분류된 이미지 개수: {len(wrong_images)}")

# 7. 시각화 (최대 5장 예시)
for i in range(min(5, len(wrong_images))):
    img = wrong_images[i].permute(1, 2, 0).numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(f"❌ Pred: {class_names[wrong_preds[i]]}, GT: {class_names[wrong_labels[i]]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
