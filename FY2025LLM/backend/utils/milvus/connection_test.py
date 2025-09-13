"""
@경로 FY2025LLM/backend/utils/milvus/connection_test.py
@실행명령어 python backend/utils/milvus/connection_test.py
"""

from pymilvus import connections

connections.connect("default", host="localhost", port="19530")
print("✅ Milvus 연결 성공!")
