# connections/elas.py

from utils.elastic.el_manager import ElasticsearchManager

# 전역 싱글톤 인스턴스
_el_manager_instance = ElasticsearchManager()

# FastAPI 의존성 주입용
def get_el() -> ElasticsearchManager:
    return _el_manager_instance
