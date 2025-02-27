from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta, timezone

class SecondBackendApi:
    def __init__(self, backend_name):
        self.backend_name = backend_name
        print("SecondBackendApi Initialized")

        # APIRouter를 클래스 내부에서 정의
        self.router = APIRouter(prefix="/second-backend-api/v1", tags=["Second Backend API"])

        # 라우트 등록
        self.router.add_api_route("/fetch/getsysdate", self.fetch_getsysdate, methods=["GET"])

    def fetch_getsysdate(self):
        # ✅ KST(한국 시간) 반환 (UTC +9)
        kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
        return JSONResponse(content={"current_datetime": kst_now.strftime("%Y-%m-%d %H:%M:%S")})

# ✅ 인스턴스 생성
secondBackendApi = SecondBackendApi(backend_name="Second")
