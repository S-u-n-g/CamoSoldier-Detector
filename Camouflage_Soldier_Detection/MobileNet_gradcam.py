import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import numpy as np
import cv2

# 1) pytorch-grad-cam 라이브러리 (1.3.6 버전)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F

def main():
    # ----------------------------
    # 1. 모델 정의 및 가중치 로드
    # ----------------------------
    model = mobilenet_v3_small(num_classes=2)      # 이진 분류용 MobileNetV3-Small
    state_dict = torch.load('mobilenetv3_epoch8_acc0.9977.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    class_names = ['camo', 'non-camo']
    # ----------------------------
    # 2. Grad-CAM 설정
    # ----------------------------
    # 마지막 convolution 레이어를 타겟으로 지정
    target_layer = model.features[-1]

    # CUDA 사용 여부 결정
    use_cuda = torch.cuda.is_available()

    # GradCAM 객체 생성 (1.3.6에서는 use_cuda 인자 사용)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)

    # ----------------------------
    # 3. 입력 이미지 전처리
    # ----------------------------
    img_path = './wrong_predictions/wrong_0038_pred-noncamouflage_gt-camouflage.png'  # 시각화할 이미지 경로
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)  # 배치 차원 추가 → (1,3,224,224)

    # 4) 모델에 통과시켜 예측 확률 계산
    with torch.no_grad():
        outputs = model(input_tensor)                           # raw logits
        probs = F.softmax(outputs, dim=1)                       # (1,2)
        pred_idx = torch.argmax(probs, dim=1).item()            # 정수 인덱스
        pred_label = class_names[pred_idx]                      # 문자열 레이블
        pred_confidence = probs[0, pred_idx].item() * 100       # 신뢰도(%)
    print(f"▶ 예측 클래스: {pred_label} ({pred_confidence:.2f}%)")


    # ----------------------------
    # 4. Grad-CAM 실행
    # ----------------------------
    # Positive 클래스(위장군인)를 인덱스 1로 가정
    target_layer = model.features[-1]               # 마지막 conv 레이어
    use_cuda = torch.cuda.is_available()
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)

    # CAM 계산 (결과: NumPy 배열 형태의 히트맵)
    grayscale_cam = cam(input_tensor, [pred_idx])[0]  # (224,224)

    # ----------------------------
    # 5. 훅 제거 (종료 시 예외 방지)
    # ----------------------------

    # ----------------------------
    # 6. 결과 시각화 및 저장
    # ----------------------------
    # 원본 이미지를 [0,1] 범위로 변환
    rgb_img = np.array(img.resize((224,224))) / 255.0
    # CAM 히트맵을 원본 위에 덧씌우기
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # BGR로 변환 후 파일로 저장
    # 이미지 위에 예측 결과 텍스트 추가 (선택)
    text = f"{pred_label}: {pred_confidence:.1f}%"
    cv2.putText(visualization, text,
                org=(10, 20),  # 위치
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 255, 255),  # 흰색
                thickness=2)

    cv2.imwrite('gradcam_with_predAIAfter3.jpg',
                (visualization[..., ::-1] * 255).astype(np.uint8))
    print("✅ gradcam_result.jpg 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()
