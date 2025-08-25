# backend/routers/gate/gate.py
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel
import os
import secrets
from datetime import timedelta
from backend.utils.log.logger import get_logger
logger = get_logger(__name__)

GATE = APIRouter(prefix="/api/gate", tags=["gate"])

# 운영자가 정한 접근 키
DEVBEAR_GATE_KEY = os.getenv("GATE_DEVBEAR_KEY")

class VerifyReq(BaseModel):
    service: str
    key: str

@GATE.post("/verify")
def verify_gate(req: VerifyReq, response: Response):
    if req.service != "devbear":
        logger.info(f"[WARNING] 알 수 없는 서비스 입니다. ")
        raise HTTPException(status_code=400, detail="알 수 없는 서비스 입니다. ")

    if req.key != DEVBEAR_GATE_KEY:
        logger.info(f"[WARNING] GATE KEY 가 일치하지 않습니다. ")
        raise HTTPException(status_code=401, detail="GATE KEY 가 일치하지 않습니다. ")

    # 간단한 세션 토큰(백엔드 보호용)
    token = secrets.token_urlsafe(24)
    max_age = int(timedelta(hours=12).total_seconds())

    # HttpOnly 쿠키(백엔드 API 보호용). 프론트 JS에서 읽을 필요 없으니 HttpOnly로 설정
    response.set_cookie(
        key="devbear_token",
        value=token,
        max_age=max_age,
        httponly=True,
        secure=False,   # HTTPS면 True 권장
        samesite="Lax",
        path="/"
    )
    return {"ok": True}
