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

    ## 1. 컬렉션 관리 ##
    def create_collection(self, name: str, dim: int, drop_if_exists: bool = True):
        """ 컬렉션 생성 (테스트 용도)"""
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
    
    def create_hr_chunks_collection(
        self,
        name: str = "hr_chunks",
        dim: int = 768,
        drop_if_exists: bool = False,
        vector_field: str = "embedding",
        metric: str = "COSINE",   # "L2" | "IP" | "COSINE"
        index_type: str = "HNSW",  # "HNSW" | "IVF_FLAT" | "IVF_SQ8" 등
    ) -> Collection:
        """ 컬렉션 생성 (HR 문서 저장)"""
        # 필요 시 기존 컬렉션 삭제
        # if drop_if_exists and name in [c.name for c in Collection.list()]:
        #     Collection(name=name).drop()

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="page_list", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_order", dtype=DataType.INT64),
            FieldSchema(name="ext", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="mime", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_uri", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="HR RAG용 문서/이미지 청크")
        col = Collection(name=name, schema=schema)

        # 벡터 인덱스 생성 (검색 성능 향상)
        if index_type.upper() == "HNSW":
            index_params = {"index_type": "HNSW", "metric_type": metric, "params": {"M": 16, "efConstruction": 200}}
        elif index_type.upper() == "IVF_FLAT":
            index_params = {"index_type": "IVF_FLAT", "metric_type": metric, "params": {"nlist": 1024}}
        else:
            # 기본값: HNSW
            index_params = {"index_type": "HNSW", "metric_type": metric, "params": {"M": 16, "efConstruction": 200}}

        col.create_index(field_name=vector_field, index_params=index_params)
        col.load()  # 검색 전에 load 필요
        print(f"컬렉션 준비 완료: {name} (dim={dim}, metric={metric}, index={index_type})")
        return col



    def insert_vectors(self, collection: Collection, ids, vectors):
        """벡터 삽입"""
        res = collection.insert([ids, vectors])
        collection.flush()
        logger.info(f"{len(ids)}개 벡터 삽입 완료")
        return res

    def search_vectors(self, collection: Collection, query_vectors, top_k: int, metric: str, nprobe: int):
        """
        벡터 검색 (metric과 nprobe를 파라미터로 선택 가능)

        Args:
            collection (Collection): 검색할 컬렉션 객체
            query_vectors (list): 검색할 벡터 리스트 (e.g. [[...]])
            top_k (int): 상위 K개 검색 (기본값 3)
            metric (str): 거리 계산 방식 ("L2", "COSINE", "IP")
            nprobe (int): 검색 정확도/속도 조절 파라미터 (클수록 정확, 느려짐)

        Returns:
            list: 검색 결과 리스트
        """
        top_k = top_k if top_k > 0 else 3
        nprobe = nprobe if nprobe > 0 else 10
        metric = metric if metric in ["L2", "COSINE", "IP"] else "L2"

        search_params = {"metric_type": metric, "params": {"nprobe": nprobe}}
        results = collection.search(
            query_vectors,
            "embedding",
            search_params,
            limit=top_k
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
    # "demo_vectors"라는 이름의 컬렉션을 만든다. 128차원짜리 벡터를 저장할 수 있다.


    ids = [1, 2, 3, 4, 5]
    vectors = np.random.rand(5, 128).tolist() # (5 x 128) 랜덤 벡터 생성
    client.insert_vectors(col, ids, vectors)  # 각 벡터에 id 1~5를 부여하고 컬렉션에 저장

    query = np.random.rand(1, 128).tolist()   # 새로운 랜덤 벡터 1개 생성
    results = client.search_vectors(col, query)
    logger.info("검색 결과:")
    for hit in results[0]:
        logger.info(f"ID={hit.id}, distance={hit.distance:.4f}")
