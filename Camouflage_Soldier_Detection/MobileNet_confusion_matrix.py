if __name__ == '__main__':
    import os
    import torch
    import torch.nn as nn
    from torchvision import models, datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # -------------------------------
    # 1. 설정
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "640mobilenetv3_epoch1_acc0.9995.pth"
    test_dir = "dataset/test"

    # -------------------------------
    # 2. 전처리
    # -------------------------------
    test_transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    class_names = test_ds.classes

    # -------------------------------
    # 3. 모델 로드
    # -------------------------------
    model = models.mobilenet_v3_small(weights=None)  # pretrained=False deprecated
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # -------------------------------
    # 4. 예측 및 결과 수집
    # -------------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # -------------------------------
    # 5. 평가지표 출력
    # -------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    print("\n[Confusion Matrix]")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names))
