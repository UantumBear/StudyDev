# base_service.py
from typing import List, Optional, Generator
from backend.utils.llm.core.types import Message
from backend.utils.azure.azure_gpt_client import AzureGPTClient

class LLMServiceBase:
    def __init__(self, client: Optional[AzureGPTClient] = None, persona=None,
                 default_temperature: float = 0.3, default_max_tokens: Optional[int] = None):
        self.client = client or AzureGPTClient()
        self.persona = persona                          # 대화에 항상 포함될 "시스템 메시지"를 제공하는 프롬프트
        self.default_temperature = default_temperature  # 기본 창의성 수치
        self.default_max_tokens = default_max_tokens    # 기본 응답 길이 제한 (None이면 클라이언트 기본값 사용)

    def _compose(self, user_messages: List[Message]) -> List[Message]:
        # 시스템 메시지(페르소나) + 사용자 메시지를 합쳐서 GPT에게 넘길 최종 메시지 리스트를 만듦.
        msgs: List[Message] = []
        if self.persona:
            msgs.extend(self.persona.system_messages())
        msgs.extend(user_messages)
        return msgs

    def send(self, messages: List[Message], **opts) -> str:
        opts.setdefault("temperature", self.default_temperature)
        opts.setdefault("max_tokens", self.default_max_tokens)
        raw = self.client.chat(self._compose(messages), **opts)
        return raw.strip()  # 공통 후처리 (필터는 자식이 오버라이드 가능)

    def send_stream(self, messages: List[Message], **opts) -> Generator[str, None, None]:
        opts.setdefault("temperature", self.default_temperature)
        opts.setdefault("max_tokens", self.default_max_tokens)
        return self.client.stream_chat(self._compose(messages), **opts)
