# app/elastic/el_manager.py
# from elasticsearch import Elasticsearch
from utils.elastic.el_client import ElClient
from config import conf

class ElManager:
    def __init__(self):
        self.es = ElClient(
            hosts=conf.ELASTICSEARCH_HOST_LOC,
            basic_auth=(conf.ELASTICSEARCH_ID_LOC, conf.ELASTICSEARCH_PW_LOC),
            verify_certs=False
        )

    def search(self, index_name: str, query: dict):
        return self.es.search(index=index_name, query=query)

    # 엘라스틱 서치의 관리
    # 커넥션 풀: 기본 내장 (HTTP keep-alive)


""" FastAPI용 Depends 주입 함수 """
el_manager = ElManager()
def get_el():
    return el_manager
