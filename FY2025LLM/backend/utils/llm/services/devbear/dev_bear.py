# FY2025LLM/backend/utils/llm/services/devbear/dev_bear.py
from typing import List, Optional, Generator
from backend.utils.llm.core.types import Message
from backend.utils.azure.azure_gpt_client import AzureGPTClient
from backend.utils.llm.prompts.base_prompt import BasePrompt
from backend.utils.llm.prompts.devbear.devbear_prompt import DevBearPrompt
from backend.utils.llm.services.llm_service_base import LLMServiceBase
from backend.utils.llm.prompts.text_filters import strip_emojis
from backend.utils.log.logger import get_logger
logger = get_logger(__name__)

class DevBear(LLMServiceBase):
    """
        DevBear 클래스는, LLMServiceBase를 상속한 '개발곰' 페르소나 전용 서비스 클래스이다.
        공통 LLM 호출 로직(temperature, max_tokens, compose 등)은 부모(Base)에서 제공
        DevBearPrompt를 기본 persona로 설정하여 캐릭터 톤/규칙/예시를 주입
        필요 시 후처리(예: 이모지 제거)를 위해 send를 오버라이드하여 필터 적용
    """
   
    def __init__(self, **kwargs):
        super().__init__(persona=DevBearPrompt(), **kwargs)

    # send 오버라이드 
    def send(self, messages, **opts) -> str:
        raw = super().send(messages, **opts)
        filtered_raw = strip_emojis(raw)
        # 개발곰은 컨셉이 Linux 화면, 따라서 윈도우 이모지 제거
        return filtered_raw
    

   