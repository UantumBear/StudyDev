from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime, timezone

# FastAPI 라우터 생성
FIRST = APIRouter(prefix="/first-backend-api/v1", tags=["First Backend API"])

class FirstBackendApi:
    def __init__(self, backend_name):
        self.backend_name = backend_name
        print("FirstBackendApi Initialized")

# 인스턴스 생성
firstBackendApi = FirstBackendApi(backend_name="First")

@FIRST.get("/fetch/getsysdate")
def fetch_getsysdate():
    # UTC(세계 표준 시간) 반환
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return JSONResponse(content={"current_datetime": utc_now})
