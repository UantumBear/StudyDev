"""
elasticsearch 테스트 파일
data 를 elastic 에 적재한다.
"""

import os
import uuid
from elasticsearch import Elasticsearch
from config import conf

# 현재 실행 파일 경로 기준 디렉토리
current_file_path = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(current_file_path)
print(CURRENT_DIR)

# 1. Elasticsearch 클라이언트 연결
es = Elasticsearch(
    hosts=conf.ELASTICSEARCH_HOST_LOC,
    basic_auth=(conf.ELASTICSEARCH_ID_LOC, conf.ELASTICSEARCH_PW_LOC),
    verify_certs=False
)

############# TXT ####################################################################
# 2. TXT 파일 경로 및 이름 설정
test_txt_name = "sample.txt"
test_txt_path = f"{CURRENT_DIR}/data/{test_txt_name}"

# 3. 텍스트 파일 읽기
with open(test_txt_path, "r", encoding="utf-8") as file:
    text = file.read()

# 4. Elasticsearch에 문서 적재
doc = {
    "file_name": test_txt_name,
    "content": text
}
# UUID를 ID로 사용해 저장
es.index(index="test_index", id=str(uuid.uuid4()), document=doc)

# 실행 명령: python utils/elastic/data_to_elastic.py
