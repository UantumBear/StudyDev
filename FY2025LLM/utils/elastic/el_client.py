from elasticsearch import Elasticsearch
from typing import Optional

""" 엘라스틱 서치의 사용자 함수들을 작성할 클래스 """

class ElClient:
    def __init__(self, hosts: str, basic_auth: tuple, verify_certs: bool = True):
        self.client = Elasticsearch(
            hosts=hosts,
            basic_auth=basic_auth,
            verify_certs=verify_certs
        )

    def ping(self) -> bool:
        return self.client.ping()

    def info(self) -> dict:
        return self.client.info()
