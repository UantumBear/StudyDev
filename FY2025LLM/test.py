import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트 추출
    :param pdf_path:
    :return:
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)

# 전체 텍스트 가져오기
extract_text_from_pdf('data/groupware/employment_rules/인사규정.pdf')

def split_text_into_chunks(text, chunk_size=500):
    """ 텍스트를 Chunk로 분리 """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_chunks(chunks):
    """ Chunk들을 임베딩으로 변환 """
    return model.encode(chunks)



def create_faiss_index(embeddings):
    """ FAISS 에 벡터 저장 """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 기반
    index.add(embeddings)
    return index

def search_faiss_index(index, query, chunks, model, top_k=5):
    """ 질의와 검색 """
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), top_k)
    results = [chunks[i] for i in I[0]]
    return results




