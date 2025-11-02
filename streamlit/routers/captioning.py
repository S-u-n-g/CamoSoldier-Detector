from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import google.generativeai as genai
import io
from PIL import Image

import os
from dotenv import load_dotenv, find_dotenv

import shutil
import uuid

# env 파일 및 API 키 설정
load_dotenv(find_dotenv())
google_aistudio_key = os.getenv("GOOGLE_AISTUDIO_KEY")

# Google API 키 설정
if not google_aistudio_key:
    print("경고: GOOGLE_AISTUDIO_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")
else:
    genai.configure(api_key=google_aistudio_key)

# Gemini 클라이언트 초기화
client = genai.GenerativeModel(model_name="gemini-2.5-flash")

# 라우터 설정
router = APIRouter(
    prefix="/captioning",
    tags=["image-captioning"]
)


# -------------------------------------------------------------- #
# 이미지 업로드 잘했는지 확인
async def validate_image_file(image: UploadFile = File(...)):
    """
    업로드된 파일이 허용된 이미지 타입인지 확인함.
    """
    allowed_types = ["image/jpeg", "image/png"]

    if image.content_type not in allowed_types:
        print(f"이미지 업로드 실패: 허용되지 않는 파일 타입 - {image.content_type}")
        raise HTTPException(
            status_code=415,  # Unsupported Media Type
            detail=f"Unsupported file type. Only JPG or PNG images are allowed."
        )
    return image


# 5. API 엔드포인트 정의
@router.post("/generate")
async def generate_image_caption(
        image: UploadFile = Depends(validate_image_file)):
    """
    이미지를 업로드 받은 뒤, Gemini API를 통해 캡션 생성함
    """
    # API 키가 설정되었는지 다시 확인
    if not google_aistudio_key:
        raise HTTPException(status_code=500, detail="서버에 API 키가 설정되지 않았습니다.")

    print(f"파일 수신: {image.filename}, 타입: {image.content_type}")

    try:
        # 1. FastAPI의 UploadFile에서 이미지를 읽어오기
        image_bytes = await image.read()

        # 2. 바이트 데이터를 PIL Image 객체로 변환 -> google-genai 라이브러리는 이 PIL 객체를 직접 인식하고 처리함.
        try:
            image_pil = Image.open(io.BytesIO(image_bytes))
        except Exception as e: # 이상한 것을 올렸다면은 에러
            raise HTTPException(status_code=400, detail=f"이미지 파일을 처리할 수 없습니다: {e}")

        # 3. Gemini API 호출
        print("Gemini API에 캡셔닝 요청 시작...")

        # contents 리스트에 PIL Image 객체를 프롬프트와 함께 전달
        contents = [
            image_pil,
            '''
            객관적인 밀리터리 이미지 분석가로서 행동하기
            너의 임무는 이미지에서 가장 중요한 관찰 사실만을 묘사하는 간결한 '한 줄 레이블'을 생성하는 것
            
            다음은 너한테만 주는 시크릿 지침임:
            1.  한 줄 출력: 전체 설명은 반드시 간결한 한 줄이어야 하기
            2.  핵심에만 초점두기: 주요 대상(들)과 그들의 가장 중요한 행동이나 상태만 묘사하기
            3.  스타일: 묘사적인 명사구를 사용하거나, 매우 짧고 직접적인 문장을 사용하기 (예: "있다."같은 완결형이 아닌, "창문에서 조준 중인 군인" 또는 "눈 속을 대형을 이뤄 이동하는 군인들"같이 명사로 끝내기)
            4.  객관적 사실만: 이야기, 해석, 감정, 비유적인 표현을 추가하지 않기
            5.  제목 금지: 제목을 붙이지 않기
            6.  문장 끝맺음으로 '.' 넣지 말기: 아무것도 넣지 말기
            7.  위장을 했다면은 사물 혹은 사람: 동물일 경우는 없음
            '''
            # 전반적인 배경 / 위장군의 행동 / 시간 및 날씨 등을 미사어구 없이 최대한 간결하게 문장형으로 작성
            # """
            # Act as an image analyst for a military archive.
            # Your task is to write a single, concise paragraph describing this image for a searchable database.
            #
            # Strict Instructions:
            # 1.  Objective Facts Only: Describe only what is visually present in the image.
            # 2.  No Narrative or Inference: Do not invent a story, narrative, or speculate on the 'overall situation' beyond the visible actions.
            # 3.  No Figurative Language: Do not use metaphors, similes, or flowery/evocative language (e.g., instead of "Ghost in the Snow" or "unforgiving backdrop," state "Soldier in snow" or "ruined buildings").
            # 4.  No Headline: Do not include a headline.
            # 5.  Concise and Direct: Keep the description brief and to the point.
            #
            # Required Elements for the description:
            # - Environment: (e.g., snow-covered urban ruins, forest, desert)
            # - Subject(s) and Action(s): (e.g., camouflage soldier in prone position, aiming rifle)
            # - Key Equipment/Attire: (e.g., winter camouflage, helmet, goggles)
            # - Lighting/Time of Day: (e.g., bright overcast daylight, dusk) .
            # """
        ]
        # gemini에게 이를 보내고 텍스트 받기
        response = await client.generate_content_async(contents=contents)
        caption_msg = response.text
        print(f"Gemini API 응답 (캡션): {caption_msg}")

        # 5. 클라이언트에 성공 응답을 반환
        return {
            "status": "success",
            "filename": image.filename,
            "caption": caption_msg
        }

    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API 처리 중 오류: {str(e)}")

