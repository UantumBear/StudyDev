from fastapi import FastAPI
from pyfiglet import Figlet
# routers
from routers.health import HEALTH
from routers.elastic.elas import ELAS
import uvicorn
from connections.el import get_el
# utils
from utils.elastic.el_server_run import start_elasticsearch_server


app = FastAPI()

# 라우터 리스트
router_list = [
    HEALTH,
    ELAS

]
# 라우터 일괄 등록
for router in router_list:
    app.include_router(router)


# 앱 시작 시점에 실행
@app.on_event("startup")
def startup_event():
    print("[STARTUP] FastAPI 시작됩니다...")
    f = Figlet(font='bubble')
    print(Figlet().getFonts())
    print(f.renderText('DevBear FY2025LLM'))

    # Elastic Server START
    start_elasticsearch_server()

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
