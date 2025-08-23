# FY2025LLM/backend/utils/llm/services/devbear/dev_bear.py
from typing import List, Optional, Generator
from backend.utils.llm.core.types import Message
from backend.utils.azure.azure_gpt_client import AzureGPTClient
from backend.utils.llm.prompts.base_prompt import BasePrompt
from backend.utils.llm.prompts.devbear_prompt import DevBearPrompt

class DevBear:
    def __init__(
        self,
        client: Optional[AzureGPTClient] = None,
        persona: Optional[BasePrompt] = None,
        default_temperature: float = 0.3,
        default_max_tokens: Optional[int] = None,
    ):
        self.client = client or AzureGPTClient()
        self.persona = persona or DevBearPrompt()
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def _compose(self, user_messages: List[Message]) -> List[Message]:
        msgs: List[Message] = []
        msgs.extend(self.persona.system_messages())
        msgs.extend(user_messages)
        return msgs

    def send(self, messages: List[Message], **opts) -> str:
        opts.setdefault("temperature", self.default_temperature)
        opts.setdefault("max_tokens", self.default_max_tokens)
        return self.client.chat(self._compose(messages), **opts)

    def send_stream(self, messages: List[Message], **opts) -> Generator[str, None, None]:
        opts.setdefault("temperature", self.default_temperature)
        opts.setdefault("max_tokens", self.default_max_tokens)
        return self.client.stream_chat(self._compose(messages), **opts)
