"""
@dir backend/run.py

"""

import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyfiglet import Figlet
# routers
from backend.routers.gate.gate import GATE
from backend.routers.health import HEALTH
from backend.routers.elastic.elas import ELAS
from backend.routers.chat.chat001 import CHAT001
from backend.routers.services.devbear.llm01 import LLM01

# utils
from backend.utils.elastic.el_server_run import start_elasticsearch_server
from backend.utils.elastic.el_manager import get_el
from backend.utils.llm.services.devbear.dev_bear import DevBear

from backend.utils.log import logger

app = FastAPI()

# 프론트 도메인만 정확히 지정 (와일드카드 X – credentials를 쓰기 때문)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,            # ← credentials: 'include' 쓰려면 필수
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 리스트
router_list = [
    GATE,
    HEALTH,
    ELAS,
    CHAT001,
    LLM01
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
    try:
        start_elasticsearch_server()
    except Exception as e:
        logger.error(str(e))

    # Elastic Client CREATE
    el = get_el()
    try:
        if el.es.ping():
            print("[STARTUP] Elasticsearch 연결 성공")
        else:
            print("[STARTUP] Elasticsearch 연결 실패")
    except Exception as e:
        print(f"[STARTUP] Elasticsearch 연결 에러: {e}")

    # ---- 서비스 인스턴스를 app.state에 저장 (의존성 주입용) ----
    app.state.devbear = DevBear()
    # app.state.hr = HRBot()


# if __name__ == "__main__":
#     uvicorn.run(
#         "run:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )


# $env:PYTHONPATH = (Get-Location).Path
# python run.py