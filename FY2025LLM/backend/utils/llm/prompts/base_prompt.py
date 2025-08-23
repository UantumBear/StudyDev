"""
@경로: FY2025LLM/backend/utils/llm/prompts/base_prompt.py
@역할:
"""

from typing import List
from backend.utils.llm.core.types import Message

class BasePrompt:
    def system_messages(self) -> List[Message]:
        return []
