# routers/llm_chat.py
from fastapi import APIRouter
from pydantic import BaseModel

CHAT001 = APIRouter()

class ChatRequest(BaseModel):
    message: str

@CHAT001.post("/chat")
def chat(req: ChatRequest):
    print(f"[CHAT] 사용자 메시지 수신: {req.message}")
    return {"reply": f"'{req.message}'에 대한 응답입니다!"}

# 이 라우터 지우기