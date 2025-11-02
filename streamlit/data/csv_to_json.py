#!/usr/bin/env python3
# run_csv_to_json.py
# 그냥 실행하면 설정에 적은 경로로 CSV를 읽어 JSON으로 저장합니다.

import csv
import json
from pathlib import Path

# ===== 설정 =====
# 입력 CSV와 출력 JSON 경로를 원하는 대로 바꾸세요.
INPUT_CSV = Path("./manifest-캡션.csv")        # 예: ./data/manifest-캡션.csv
OUTPUT_JSON = Path("./detections_from_manifest.json")  # 예: ./data/..json
CSV_ENCODING = "utf-8-sig"  # 엑셀로 만든 CSV면 utf-8-sig가 안전. 일반 UTF-8이면 "utf-8"
# =================

KEEP_FIELDS = ["id", "timestamp", "lat", "lon", "score", "caption"]


def to_float_or_pass(v: str):
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        # 숫자 변환이 안 되면 원문 유지(데이터 오류 대비)
        return v


def row_to_minimal(row: dict) -> dict:
    out = {}
    for f in KEEP_FIELDS:
        if f not in row:
            raise KeyError(f"CSV에 '{f}' 헤더가 없습니다. 실제 헤더: {list(row.keys())}")
        val = row[f]
        if isinstance(val, str):
            val = val.strip()
        if f in ("lat", "lon", "score"):
            out[f] = to_float_or_pass(val)
        else:
            out[f] = val
    return out


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"입력 CSV 파일을 찾을 수 없습니다: {INPUT_CSV.resolve()}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with INPUT_CSV.open("r", encoding=CSV_ENCODING, newline="") as f:
        # 구분자 자동 추정(콤마가 기본이지만 안전하게)
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(f)

        # 헤더 검증
        if reader.fieldnames is None:
            raise ValueError("CSV에서 헤더를 읽지 못했습니다. 첫 줄에 헤더가 있는지 확인하세요.")
        missing = [h for h in KEEP_FIELDS if h not in reader.fieldnames]
        if missing:
            raise KeyError(f"CSV 헤더에 다음 필드가 없습니다: {missing}\n실제 헤더: {reader.fieldnames}")

        for row in reader:
            # 완전 공백 행은 스킵
            if not any((row.get(k) or "").strip() for k in KEEP_FIELDS):
                continue
            records.append(row_to_minimal(row))

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"완료: {len(records)}개 레코드를 '{OUTPUT_JSON}'에 저장했습니다.")


if __name__ == "__main__":
    main()
