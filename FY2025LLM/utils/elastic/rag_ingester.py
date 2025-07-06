#
"""
@경로 utils/elastic/rag_ingestor.py
@역할 임베딩된 데이터를 elasticsearch 에 적재하는 역할
"""
import uuid
from elasticsearch import Elasticsearch
from config import conf
from typing import List, Dict


class RAGIngestor:
    def __init__(self, index_name: str = "test_index"):
        self.index_name = index_name
        self.es = Elasticsearch(
            hosts=conf.ELASTICSEARCH_HOST_LOC,
            basic_auth=(conf.ELASTICSEARCH_ID_LOC, conf.ELASTICSEARCH_PW_LOC),
            verify_certs=False # 인증서 검사 안 함
        )
        self._ensure_index()



    def _ensure_index(self):
        """
        Elasticsearch에 인덱스가 없으면 새로 생성
        """
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                mappings={
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": 768}
                    }
                }
            )
            print(f"인덱스 '{self.index_name}' 생성 완료.")
        else:
            print(f"인덱스 '{self.index_name}' 이미 존재.")

    def ingest_embeddings(self, data: List[Dict[str, object]]):
        """
        임베딩된 데이터 (텍스트 + 벡터) 리스트를 Elasticsearch에 적재
        """
        for item in data:
            doc = {
                "text": item["text"],
                "embedding": item["embedding"]
            }
            self.es.index(index=self.index_name, id=str(uuid.uuid4()), document=doc)

        print(f"[SUCCESS] 총 {len(data)}건 Elasticsearch에 적재 완료.")


# 테스트용 메인 실행 예시
if __name__ == "__main__":
    from utils.embeddings.embedder import Embedder
    import os

    embedder = Embedder()
    file_path = os.path.join(conf.PROJECT_ROOT_DIRECTORY, "data/test/test.txt")
    data = embedder.embed_file(file_path)

    ingestor = RAGIngestor(index_name="test_index")

    # 기존 인덱스 삭제 (옵션)
    ingestor.es.indices.delete(index=ingestor.index_name)
    # 새로운 인덱스 생성 (옵션)
    ingestor._ensure_index()

    ingestor.ingest_embeddings(data)

# python utils/elastic/rag_ingester.py