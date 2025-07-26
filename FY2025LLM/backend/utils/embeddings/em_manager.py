from backend.utils.embeddings.embedder import Embedder
from typing import List, Dict

class EmManager:
    """
    앱 전역에서 공유할 Embedder 객체를 래핑한 싱글톤 관리자
    """
    def __init__(self):
        self.embedder = Embedder()


""" FastAPI 주입 용도 """
em_manager = EmManager()
def get_em_manager() -> EmManager:
    return em_manager