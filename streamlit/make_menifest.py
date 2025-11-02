import csv, os, random
from datetime import datetime, timedelta, timezone  
import numpy as np

# 이미지 폴더 경로
img_dir = "./result"
output_csv = "./manifest.csv"

# 캡션 후보
captions = [
    "수풀 속 위장 병력 감지",
    "은폐된 병사 포착",
    "나무 옆에서 위장된 인원 탐지",
    "위장망 근처에 위치한 병력",
    "초원 위에 엎드린 병사",
    "숲속에서 관측 장비를 든 인원"
]

# 기본 좌표 (임의 중심점)
base_lat, base_lon = 36.35, 127.30

# 이미지 파일명 목록 가져오기
files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

# CSV 생성
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "filename", "timestamp", "lat", "lon", "score", "class", "caption"])
    now = datetime.now(timezone.utc)
    for i, file in enumerate(files):
        img_id = file.split(".")[0]
        ts = (now - timedelta(minutes=random.randint(0, 60 * 24 * 14))).isoformat()
        lat = base_lat + np.random.normal(0, 0.01)
        lon = base_lon + np.random.normal(0, 0.01)
        score = max(0, min(1, np.random.beta(5, 2)))
        caption = random.choice(captions)
        writer.writerow([img_id, file, ts, lat, lon, score, "soldier", caption])

print(f"✅ manifest.csv 생성 완료 ({len(files)}개)")
