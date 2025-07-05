"""
텍스트 -> 임베딩 삽입
"""

class EmbeddingHandler:
    def __init__(self, es_client, index_name):
        self.es_client = es_client
        self.index_name = index_name

    def insert_embeddings(self, texts: List[str], metadata: dict):
        for i, text in enumerate(texts):
            doc = {
                "content": text,
                "metadata": metadata
            }
            self.es_client.index(index=self.index_name, document=doc)
