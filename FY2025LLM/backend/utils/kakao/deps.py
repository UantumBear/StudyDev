# @경로: backend/utils/kakao/deps.py
from functools import lru_cache
from backend.utils.kakao.client import KakaoClient

@lru_cache(maxsize=1)
def _kakao_singleton() -> KakaoClient:
    # conf 검증은 KakaoClient.__init__ 내부에서 수행
    return KakaoClient()

def get_kakao_client() -> KakaoClient:
    """FastAPI Depends용 주입 함수"""
    return _kakao_singleton()
