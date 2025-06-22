import pdfplumber
from elasticsearch import Elasticsearch
import uuid
from config import conf
import os

# 실행 파일 경로
current_file_path = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(current_file_path)
print(CURRENT_DIR)

# 1. Elasticsearch 클라이언트 연결
es = Elasticsearch(hosts=conf.ELASTICSEARCH_HOST_LOC, basic_auth=(conf.ELASTICSEARCH_ID_LOC, conf.ELASTICSEARCH_PW_LOC), verify_certs=False)

############# PDF ####################################################################
# 2. PDF → 텍스트 추출
test_pdf_name = "테스트01.pdf"
test_pdf_path = f"{CURRENT_DIR}/data/{test_pdf_name}"
with pdfplumber.open(test_pdf_path) as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

# 3. Elasticsearch에 문서 적재
doc = {
    "file_name": test_pdf_name,
    "content": text
}
# UUID를 ID로 사용해 저장
es.index(index="pdf_index", id=str(uuid.uuid4()), document=doc)




# python utilities/elastic/testrun.py

