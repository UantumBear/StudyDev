"""
@경로 FY2025LLM/backend/schemas/llm_chat.py
chating 용 Schema
"""

from pydantic import BaseModel
from typing import List
from backend.utils.llm.core.types import Message

class ChatReq(BaseModel):
    message: List[Message]

class AskReq(BaseModel):
    prompt: str