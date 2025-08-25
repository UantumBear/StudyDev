"""
@경로: FY2025LLM/backend/routers/services/devbear/llm01.py
"""
from typing import List
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas.llm_chat import ChatReq, AskReq
from backend.utils.llm.core.types import Message
from backend.utils.llm.services.devbear.dev_bear import DevBear
from backend.utils.log.logger import get_logger
logger = get_logger(__name__)
LLM01 = APIRouter(
    prefix="/services/devbear",
    tags=["DevBear LLM"],
)

# 앱에 저장된 서비스 인스턴스를 DI로 꺼내는 의존성
def get_devbear(request: Request) -> DevBear:
    # run.py에서 app.state.devbear = DevBear() 로 세팅되어 있어야 합니다.
    devbear = getattr(request.app.state, "devbear", None)
    if devbear is None:
        logger.error("DevBear 서비스가 초기화되지 않았습니다.")
        raise HTTPException(status_code=500, detail="DevBear 서비스가 초기화되지 않았습니다.")
    return devbear


@LLM01.post("/chat")
def chat(req: ChatReq, devbear: DevBear = Depends(get_devbear)):
    """
    다중 메시지(대화 이력)로 호출
    body: { "messages": [ { "role": "user", "content": "..." }, ... ] }
    """
    try:
        return {"reply": devbear.send(req.messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 오류: {e}")


@LLM01.post("/ask")
def ask(req: AskReq, devbear: DevBear = Depends(get_devbear)):
    """
    단일 프롬프트로 호출
    body: { "prompt": "사용자 입력" }
    """
    try:
        # DevBear는 send(messages)만 제공하므로 prompt를 메시지로 감싸서 호출
        msgs: List[Message] = [{"role": "user", "content": req.prompt}]
        return {"reply": devbear.send(msgs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 오류: {e}")


@LLM01.post("/chat/stream")
def chat_stream(req: ChatReq, devbear: DevBear = Depends(get_devbear)):
    """
    (옵션) 스트리밍 응답. 클라이언트가 텍스트를 조각 단위로 받도록 할 때 사용.
    """
    try:
        def gen():
            for chunk in devbear.send_stream(req.messages):
                # 단순 텍스트 스트리밍 (필요 시 SSE 포맷으로 변경 가능)
                yield chunk
        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 스트리밍 오류: {e}")
