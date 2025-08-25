"""
@경로 FY2025LLM/utils/azure/azure_gpt_client.py
@실행예시 python -m backend.utils.azure.azure_gpt_client :: 모듈경로를 이해하고 실행
"""
# 파일: backend/utils/azure/azure_gpt_client.py
from __future__ import annotations

import os
import time
from typing import List, Dict, Optional, Iterable, Generator, Literal, Union

from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import BaseModel
from config import conf


Role = Literal["system", "user", "assistant", "tool"]
Message = Dict[str, str]


class AzureGPTConfig(BaseModel):
    """런타임에 .env/conf에서 읽어온 설정을 한 곳에 모은다."""
    api_key: str
    endpoint: str
    api_version: str
    deployment: str
    request_timeout: float = 60.0            # 초
    max_retries: int = 3
    initial_backoff: float = 0.5             # 초

    @classmethod
    def from_conf(cls) -> "AzureGPTConfig":
        return cls(
            api_key=conf.AZURE_GPT_MODEL_KEY,
            endpoint=conf.AZURE_GPT_MODEL_ENDPOINT,
            api_version=conf.AZURE_GPT_MODEL_API_VERSION,
            deployment=conf.AZURE_GPT_MODEL_NAME,
        )


class AzureGPTClient:
    """
    Azure OpenAI Chat Completions 전용 클라이언트.
    - 동기 방식 chat(), 스트리밍 stream_chat() 제공
    - 지수 백오프 재시도
    - 필요한 필드만 파싱(스키마-관대한 파싱)
    """

    def __init__(self, cfg: Optional[AzureGPTConfig] = None):
        self.cfg = cfg or AzureGPTConfig.from_conf()
        self.client = AzureOpenAI(
            api_key=self.cfg.api_key,
            azure_endpoint=self.cfg.endpoint,
            api_version=self.cfg.api_version,
            timeout=self.cfg.request_timeout,   # openai>=1.0 계열에서 지원
        )

    # ---- 내부: 재시도 래퍼 -------------------------------------------------
    def _with_retry(self, fn, *args, **kwargs):
        backoff = self.cfg.initial_backoff
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (RateLimitError, APITimeoutError) as e:
                if attempt == self.cfg.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2
            except APIError as e:
                # 5xx만 재시도, 4xx는 즉시 전파
                if getattr(e, "status_code", None) and 500 <= e.status_code < 600:
                    if attempt == self.cfg.max_retries:
                        raise
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise

    # ---- 공개: 단발성 응답 -------------------------------------------------
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> str:
        """
        반환: 어시스턴트 최종 텍스트(content)
        """
        resp = self._with_retry(
            self.client.chat.completions.create,
            model=self.cfg.deployment,              # 배포 이름
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        # 필요한 필드만 사용(스키마-관대한 파싱)
        return resp.choices[0].message.content if resp and resp.choices else ""

    # ---- 공개: 스트리밍 -----------------------------------------------------
    def stream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.18,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Generator[str, None, None]:
        """
        스트리밍 청크를 yield 합니다. (FastAPI SSE/웹소켓에 연결해서 쓰기 좋음)
        """
        stream = self._with_retry(
            self.client.chat.completions.create,
            model=self.cfg.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **extra,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
            if delta:
                yield delta

    # ---- 헬퍼 --------------------------------------------------------------
    def simple_prompt(self, prompt: str, system: Optional[str] = None) -> str:
        msgs: List[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs)


# ---- 단독 실행 테스트 -------------------------------------------------------
if __name__ == "__main__":
    client = AzureGPTClient()
    answer = client.simple_prompt("안녕, 네 이름이 뭐야?", system="You are a helpful assistant.")
    print(answer)




# 초기 테스트 소스
# from openai import AzureOpenAI
# from config import conf
#
# client = AzureOpenAI(
#     api_key=conf.AZURE_GPT_MODEL_KEY,
#     azure_endpoint=conf.AZURE_GPT_MODEL_ENDPOINT,
#     api_version=conf.AZURE_GPT_MODEL_API_VERSION
# )
#
# resp = client.chat.completions.create(
#     model=conf.AZURE_GPT_MODEL_NAME,  # 배포 이름 그대로
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "안녕"}
#     ]
# )
#
# print(resp.choices[0].message.content)
