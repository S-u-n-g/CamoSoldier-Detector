# searching.py (캡션 검색 전용 서비스로 수정됨)

from fastapi import (
    APIRouter, Query, HTTPException, Depends,
    UploadFile, File, Body, Form
)
from elasticsearch import Elasticsearch, NotFoundError
from typing import List, Dict, Any, Optional
import logging
import json

# --- 1. ES 클라이언트 및 기본 설정 ---
try:
    es = Elasticsearch("http://localhost:9200")
    es.ping()
    logging.info("Elasticsearch에 성공적으로 연결되었습니다.")
except Exception as e:
    logging.error(f"Elasticsearch 연결 실패: {e}")
    raise RuntimeError(f"Elasticsearch 연결 실패: {e}")

# (수정) 캡션 검색 전용 인덱스 이름
INDEX_NAME = "camo_caption_search"

# --- 2. 라우터 생성 ---
router = APIRouter()

# --- 3. Elasticsearch 인덱스 설정 함수 (수정) ---
def setup_elasticsearch():
    """
    (수정) 캡션 검색 전용 인덱스를 생성합니다.
    'id'는 ES의 '_id'를 사용하고, 'caption'과 'filename'만 저장합니다.
    """

    # 3-1. 인덱스 설정 (nori 분석기)
    settings = {
        "analysis": {
            "analyzer": {
                "nori_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer"
                }
            }
        }
    }

    # 3-2. 매핑 설정 (수정: caption과 filename만)
    mappings = {
        "properties": {
            "filename": {"type": "keyword"},  # (요청사항) id, filename, caption
            "caption": {
                "type": "text",
                "analyzer": "nori_analyzer"  # 캡션만 nori로 분석
            },
            # lat, lon, timestamp, score, class, extra 등 모두 제거
        }
    }

    try:
        if not es.indices.exists(index=INDEX_NAME):
            es.indices.create(
                index=INDEX_NAME,
                settings=settings,
                mappings=mappings
            )
            print(f"인덱스 '{INDEX_NAME}' 생성 완료 (캡션 검색 전용).")
        else:
            es.indices.put_mapping(index=INDEX_NAME, properties=mappings["properties"])
            print(f"인덱스 '{INDEX_NAME}'가 이미 존재합니다. 매핑을 업데이트했습니다.")

    except Exception as e:
        print(f"Elasticsearch 인덱스 설정 중 오류 발생: {e}")
        raise

# --- 4. 캡션 검색 API (유지) ---
@router.get("/search")
def search_captions(
        q: str = Query(..., min_length=2, description="검색어"),
        limit: int = Query(200, description="최대 반환 개수")
):
    """
    Elasticsearch('match' 쿼리)를 사용하여
    검색어와 캡션 키워드가 유사한 문서 목록을 반환합니다.
    (수정: id와 sim_score만 반환)
    """
    print(f"캡션 검색 요청: '{q}'")

    es_query = {
        "match": {
            "caption": {
                "query": q,
                "analyzer": "nori_analyzer"
            }
        }
    }

    try:
        response = es.search(
            index=INDEX_NAME,
            query=es_query,
            size=limit,
            min_score=0.1
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ES 검색 오류: {e}")

    # (수정) 전체 문서(_source) 대신 'id'와 '_score'만 반환
    results = []
    for hit in response['hits']['hits']:
        results.append({
            "id": hit["_id"],  # ES의 _id (메인 DB의 id와 동일)
            "_sim_score": hit['_score']
        })

    return results

# --- 5. (신규) ES 캡션 색인 동기화 API (Create/Update) ---
@router.post("/sync")
async def sync_document(
        id: str = Form(..., description="메인 DB의 고유 ID (ES의 _id로 사용됨)"),
        caption: str = Form(..., description="색인할 캡션 텍스트"),
        filename: str = Form(..., description="참조용 파일명")
):
    """
    메인 DB에서 C/U가 발생했을 때 호출됩니다.
    ES의 캡션 검색 인덱스에 데이터를 생성하거나 덮어씁니다.
    """
    try:
        document = {
            "caption": caption,
            "filename": filename
        }
        # id를 ES의 _id로 사용하여 문서를 '색인(index)' (없으면 생성, 있으면 덮어쓰기)
        es.index(index=INDEX_NAME, id=id, document=document)
        es.indices.refresh(index=INDEX_NAME) # 즉시 검색 가능하도록 refresh
        return {
            "status": "success",
            "message": f"ES 캡션 인덱스 동기화 완료 (ID: {id})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ES 동기화 오류: {e}")

# --- 6. (신규) ES 캡션 색인 동기화 API (Delete) ---
@router.delete("/sync/{det_id}")
async def delete_document_from_index(det_id: str):
    """
    메인 DB에서 D가 발생했을 때 호출됩니다.
    ES의 캡션 검색 인덱스에서 문서를 삭제합니다.
    """
    try:
        response = es.delete(index=INDEX_NAME, id=det_id)
        es.indices.refresh(index=INDEX_NAME)
        return {
            "status": "success",
            "message": f"ES 캡션 인덱스 삭제 완료 (ID: {det_id})"
        }
    except NotFoundError:
        # 이미 없어도 성공으로 처리
        return {"status": "success", "message": f"ES에 이미 존재하지 않음 (ID: {det_id})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ES 삭제 중 오류: {e}")


# --- 7. (제거) 기존 GET/POST/DELETE /detections 엔드포인트 ---
# (이 서비스는 더 이상 메인 데이터베이스가 아니므로 모두 제거)


# --- 8. 메인 실행 (인덱스 설정) ---
if __name__ == "__main__":
    print("Elasticsearch 캡션 검색 인덱스 설정을 시작합니다...")
    setup_elasticsearch()

    # (수정) 샘플 데이터 인덱싱 (캡션과 파일명만)
    SAMPLE_CAPTIONS = [
        {"id": "result_000001", "caption": "숲속에서 위장복을 입고 소총을 든 두 명의 군인", "filename": "s1.jpg"},
        {"id": "result_000002", "caption": "사막에서 기동하는 탱크와 모래 먼지", "filename": "t1.jpg"},
        {"id": "result_000003", "caption": "폐허가 된 건물 창가에 저격수가 위장한 채 숨어있다", "filename": "s2.jpg"},
    ]
    print("샘플 캡션 데이터 색인 시작...")
    for item in SAMPLE_CAPTIONS:
        try:
            doc_id = item.pop("id")
            es.index(index=INDEX_NAME, id=doc_id, document=item)
        except Exception as e:
            print(f"ID {doc_id} 색인 오류: {e}")

    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)['count']
    print(f"'{INDEX_NAME}'에 총 {count}개의 캡션 문서가 저장되었습니다.")