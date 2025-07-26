"""
@경로 utils.embeddings.model_downloader.py
@역할 임베딩 모델 다운로드
"""

from sentence_transformers import SentenceTransformer

model_name = "BM-K/KoSimCSE-roberta"
save_path = "models/BM-K/KoSimCSE-roberta"  # 원하는 저장 경로

# 1. 모델 로드 (최초 1회 인터넷 필요)
model = SentenceTransformer(model_name)

# 2. 로컬 저장
model.save(save_path)

print("모델이 로컬에 저장되었습니다.")

# python utils/embeddings/model_downloader.py