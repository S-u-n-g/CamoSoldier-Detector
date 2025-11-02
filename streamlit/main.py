import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

# --- 1. 라우터 임포트 ---
# (참고: routers 폴더에 captioning.py, searching.py, reporting.py가 모두 있어야 함)
from routers import captioning, searching, reporting

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="통합 위장군 탐지 API",
    description="캡션 생성, ES 검색, 보고서 생성 기능을 모두 제공"
)

# --- 'static' 폴더 (이미지 저장용) ---
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 'out' 폴더 (보고서 저장용) ---
if not os.path.exists("out"):
    os.makedirs("out")
app.mount("/files", StaticFiles(directory="out"), name="files")

# --- 라우터 포함 ---

# 1. 캡션 생성 라우터 (e.g., /captioning/generate)
app.include_router(captioning.router)

# 2. ES 검색 라우터 (e.g., /searching/search, /searching/sync)
app.include_router(
    searching.router,
    prefix="/searching",
    tags=["Image Searching (ES)"]
)

# 3. 보고서 생성 라우터 (e.g., /generate, /health)
app.include_router(
    reporting.router,
    tags=["Report Generation (LLM)"]
)


# --- 5. 루트 엔드포인트 ---
@app.get("/")
def read_root():
    return {
        "message": "통합 API 서버 실행 중",
        "docs_url": "/docs"
    }


# --- uvicorn 실행을 위한 메인 ---
if __name__ == "__main__":
    print("--- 개발 서버를 http://127.0.0.1:8000 에서 시작합니다 ---")
    print("--- API 문서: http://127.0.0.1:8000/docs ---")

    # (참고) searching.py의 setup_elasticsearch()를 서버 시작 시 호출
    try:
        searching.setup_elasticsearch()
    except Exception as e:
        print(f"경고: Elasticsearch 인덱스 설정 실패 (ES 서버가 실행 중인지 확인하세요): {e}")

    uvicorn.run(app, host="127.0.0.1", port=8000)
