"""
@경로: FY2025LLM/backend/routers/services/devbear/llm01.py
"""

from typing import List
from fastapi import APIRouter, Depends, Request
from backend.schemas.llm_chat import  ChatReq, AskReq
from backend.utils.llm.services.devbear.dev_bear import DevBear

# 라우터 객체를 전역으로 선언
LLM01 = APIRouter(
    prefix="/services/devbear",   # 최종 엔드포인트 prefix
    tags=["DevBear LLM"],         # Swagger UI 구분용 태그
)

# 앱에 저장된 서비스 인스턴스를 DI로 꺼내는 의존성
def get_devbear(request: Request) -> DevBear:
    return request.app.state.devbear  # run.py에서 세팅할 예정


@LLM01.post("/chat")
def chat(req: ChatReq, devbear: DevBear = Depends(get_devbear)):
    return {"reply": devbear.send(req.messages)}

@LLM01.post("/ask")
def ask(req: AskReq, devbear: DevBear = Depends(get_devbear)):
    return {"reply": devbear.ask(req.prompt)}