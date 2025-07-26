"""
@dir backend/run.py

"""

import uvicorn
import os
from fastapi import FastAPI
from pyfiglet import Figlet
# routers
from backend.routers.health import HEALTH
from backend.routers.elastic.elas import ELAS
from backend.routers.chat.chat001 import CHAT001
# utils
from backend.utils.elastic.el_server_run import start_elasticsearch_server
from backend.utils.elastic.el_manager import get_el


app = FastAPI()

# 라우터 리스트
router_list = [
    HEALTH,
    ELAS,
    CHAT001

]
# 라우터 일괄 등록
for router in router_list:
    app.include_router(router)


# 앱 시작 시점에 실행
@app.on_event("startup")
def startup_event():
    print("[STARTUP] FastAPI 시작됩니다...")
    f = Figlet(font='soft')
    # print(Figlet().getFonts())
    print(f.renderText('DevBear FY2025LLM'))
    print('본 프로젝트는 개발곰이 2025년 동안 공부한 내용을 정리하고자 하는 프로젝트입니다!')

    # Elastic Server START :: 로컬 실행 용도, Docker 환경에서는 compose.yml 에서 실행
    # start_elasticsearch_server()

    # Elastic Client CREATE
    el = get_el()
    try:
        if el.es.ping():
            print("[STARTUP] Elasticsearch 연결 성공")
        else:
            print("[STARTUP] Elasticsearch 연결 실패")
    except Exception as e:
        print(f"[STARTUP] Elasticsearch 연결 에러: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


# $env:PYTHONPATH = (Get-Location).Path
# python run.py