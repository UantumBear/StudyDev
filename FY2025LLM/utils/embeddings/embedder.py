"""
@경로 utils/embeddings/embedder.py
@역할 txt 파일의 벡터화를 수행한다.
"""

import os
from sentence_transformers import SentenceTransformer
from typing import List, Union
from config import conf

class Embedder:
    def __init__(self, model_path: Union[str, os.PathLike] = f"{conf.PROJECT_ROOT_DIRECTORY}/models/BM-K/KoSimCSE-roberta"):
        """
        로컬 디렉토리에서 사전 다운로드된 임베딩 모델을 로드한다.
        """
        self.model_path = model_path
        self.model = SentenceTransformer(model_path)
        print(f"로컬 모델 로딩 완료: {model_path}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        문자열 리스트를 입력받아 임베딩 벡터 리스트를 반환한다.
        """
        if not texts:
            raise ValueError("입력 텍스트가 비어 있습니다.")

        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_file(self, file_path: str, encoding: str = "utf-8") -> List[dict]:
        """
        텍스트 파일을 임베딩하고, 텍스트와 벡터를 딕셔너리 형태로 반환하한다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

        with open(file_path, "r", encoding=encoding) as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"{len(lines)} 문장 임베딩 중...")
        embeddings = self.embed_texts(lines)

        return [{"text": line, "embedding": emb} for line, emb in zip(lines, embeddings)]


if __name__ == "__main__":
    embedder = Embedder()
    data_path = f"{conf.PROJECT_ROOT_DIRECTORY}/data/test/test.txt"
    results = embedder.embed_file(data_path)

    for item in results[:3]:  # 상위 3개만 출력
        print("Text:", item["text"])
        print("Embedding:", item["embedding"][:5], "...")  # 벡터 일부만 출력
        print()

# python utils/embeddings/embedder.py