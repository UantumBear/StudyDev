"""
@경로: backend/utils/milvus/milvus_client.py
@실행: python backend/utils/milvus/milvus_client.py
"""
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType
import threading
from backend.utils.log.logger import get_logger
logger = get_logger(__name__)

class MilvusClient:
    """Milvus 연결 및 컬렉션 관리용 싱글톤 클라이언트"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, host="localhost", port="19530"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_connection(host, port)
            return cls._instance

    def _init_connection(self, host, port):
        self.host = host
        self.port = port
        connections.connect("default", host=self.host, port=self.port)
        logger.info(f"Milvus 연결 성공 (host={self.host}, port={self.port})")

    def create_collection(self, name: str, dim: int, drop_if_exists: bool = True):
        """컬렉션 생성 (필요 시 기존 것 삭제)"""
        if drop_if_exists and name in [c.name for c in Collection.list()]:
            Collection(name=name).drop()
            logger.info(f" 기존 컬렉션 {name} 삭제됨")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="기본 벡터 컬렉션")
        collection = Collection(name=name, schema=schema)
        logger.info(f"컬렉션 생성 완료: {name}")
        return collection

    def insert_vectors(self, collection: Collection, ids, vectors):
        """벡터 삽입"""
        res = collection.insert([ids, vectors])
        collection.flush()
        logger.info(f"{len(ids)}개 벡터 삽입 완료")
        return res

    def search_vectors(self, collection: Collection, query_vectors, top_k=3):
        """벡터 검색"""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            query_vectors, "embedding", search_params, limit=top_k
        )
        return results

    def drop_collection(self, name: str):
        """컬렉션 삭제"""
        if name in [c.name for c in Collection.list()]:
            Collection(name=name).drop()
            logger.info(f"컬렉션 {name} 삭제 완료")


# 테스트 코드
if __name__ == "__main__":
    import numpy as np

    client = MilvusClient()
    col = client.create_collection("demo_vectors", dim=128)

    ids = [1, 2, 3, 4, 5]
    vectors = np.random.rand(5, 128).tolist()
    client.insert_vectors(col, ids, vectors)

    query = np.random.rand(1, 128).tolist()
    results = client.search_vectors(col, query)
    logger.info("검색 결과:")
    for hit in results[0]:
        logger.info(f"ID={hit.id}, distance={hit.distance:.4f}")
