"""
@경로: FY2025LLM/backend/routers/services/devbear/llm.py
"""

from typing import List
from fastapi import APIRouter, Depends
from backend.schemas.llm_chat import  ChatReq, AskReq

# 라우터 객체를 전역으로 선언
LLM01 = APIRouter(
    prefix="/services/devbear",   # 최종 엔드포인트 prefix
    tags=["DevBear LLM"],         # Swagger UI 구분용 태그
)


def make_chat_router(service, *, prefix: str, tag: str) -> APIRouter:
    """
    service: DevBear/HR 등 send(), send_stream(), ask()를 가진 객체
    """
    router = APIRouter(prefix=prefix, tags=[tag])

    @router.post("/chat")
    def chat(req: ChatReq):
        return {"reply": service.send(req.messages)}

    @router.post("/ask")
    def ask(req: AskReq):
        return {"reply": service.ask(req.prompt)}

    return router