"""
@경로: FY2025LLM/backend/utils/llm/core/types.py
@역할:
"""

from typing import TypedDict, Literal
Role = Literal["system", "user", "assistant", "tool"]
class Message(TypedDict):
    role: Role
    content: str
