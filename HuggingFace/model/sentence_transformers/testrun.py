from sentence_transformers import SentenceTransformer, util
import torch

# Bi-Encoder 모델 로드
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # 영어
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2') # 다국어 모델

# 현재 디바이스 확인
print("현재 사용 디바이스:", "GPU" if torch.cuda.is_available() else "CPU")

# 모델 로컬 저장
# model.save('./model/all-MiniLM-L6-v2')  # 로컬 디렉토리에 저장

# 쿼리와 문서들
query = "내가 챗봇을 개발하려면 어떻게 해야 할까?"
documents = [
    "챗봇을 개발하려면 어떻게 해야 할까?",
    "내가 챗봇을 개발하려면 어떻게 해야 할까?"
]

# 임베딩 변환
query_embedding = model.encode(query, convert_to_tensor=True)
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# 유사도 계산
similarities = util.cos_sim(query_embedding, doc_embeddings)

# 결과 출력
for i, score in enumerate(similarities[0]):
    print(f"문서 {i+1}: {documents[i]} (유사도: {score:.4f})")
