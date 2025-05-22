import requests
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential  # API Key 인증 방식 사용



# Step 0. 환경변수 로드

import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential  # API Key 인증 방식

# ✅ 환경 변수 로드
load_dotenv(dotenv_path=".env_azure")

# ✅ .env 파일에서 API 설정 가져오기
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# ✅ 디버깅: 환경 변수 값 출력
print(f"Endpoint: {endpoint}")
print(f"API Key: {'EXISTS' if api_key else 'MISSING'}")  # 보안상 실제 키는 출력하지 않음
print(f"Deployment Name: {deployment_name}")
print(f"API Version: {api_version}")

# ✅ 환경 변수 체크 (문제 발생 시 오류 출력)
if not all([endpoint, api_key, deployment_name, api_version]):
    raise ValueError("환경 변수가 올바르게 설정되지 않았습니다. .env 파일을 확인하세요.")

# ✅ API Key 인증 방식 사용
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
)

# ✅ API 요청 실행
response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?"),
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment_name  # ✅ 올바른 배포 모델 이름 적용
)

# ✅ 응답 출력
print(response.choices[0].message.content)
