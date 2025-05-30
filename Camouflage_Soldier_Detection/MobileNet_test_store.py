import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 1. 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./mobilenetv3_epoch8_acc0.9977.pth"
test_dir = "./dataset/test"  # test/camouflage, test/noncamouflage
save_wrong_dir = "./wrong_predictions"  # 저장할 폴더

# 저장 폴더 생성
os.makedirs(save_wrong_dir, exist_ok=True)

# 2. 전처리
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 데이터셋 & 로더
test_ds = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
class_names = test_ds.classes  # ['camouflage', 'noncamouflage']

# 4. 모델 로드
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 5. 오분류 저장
wrong_count = 0
image_index = 0  # 이미지 고유 인덱스

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        labels = labels.cpu()

        for i in range(len(images)):
            image_index += 1
            if preds[i] != labels[i]:
                wrong_count += 1
                # 저장할 파일 이름 예시: wrong_0003_pred-noncamouflage_gt-camouflage.png
                pred_label = class_names[preds[i].item()]
                true_label = class_names[labels[i].item()]
                file_name = f"wrong_{image_index:04d}_pred-{pred_label}_gt-{true_label}.png"
                file_path = os.path.join(save_wrong_dir, file_name)

                # 정규화 해제 후 저장
                unnorm = images[i].cpu().clone()
                for t, m, s in zip(unnorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                    t.mul_(s).add_(m)
                unnorm = torch.clamp(unnorm, 0, 1)

                save_image(unnorm, file_path)

print(f"✅ 오분류 이미지 {wrong_count}장을 '{save_wrong_dir}' 폴더에 저장 완료했습니다.")
