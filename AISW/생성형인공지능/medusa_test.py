# 필요한 라이브러리 설치
# !pip install transformers accelerate torch

# 필요한 모듈 import
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 이름 지정
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
model_download_path = "data/model"

# 모델 이름과 로컬 저장 경로 설정
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
local_model_path = "data/model"  # 로컬 저장 경로

# 모델 경로에 필요한 파일이 존재하는지 확인하는 함수
def is_model_downloaded(path):
    # 필요한 주요 파일이 모두 존재하는지 확인
    required_files = [
        os.path.join(path, "config.json"),
        os.path.join(path, "pytorch_model.bin"),
        os.path.join(path, "tokenizer.json")
    ]
    return all(os.path.exists(file) for file in required_files)

# 모델과 토크나이저 전역적으로 선언
tokenizer = None
model = None

# 모델과 토크나이저 로드 또는 다운로드
if is_model_downloaded(local_model_path):
    print("모델이 로컬에 이미 존재합니다. 로드합니다.")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
else:
    print("로컬에 모델이 없습니다. 다운로드 후 저장합니다.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"모델과 토크나이저가 '{local_model_path}'에 저장되었습니다.")


# 입력 텍스트 리스트
input_texts = [
    "안녕하세요! 오늘의 날씨는 어떨까요?",
    "GPT 모델은 어떤 원리로 동작하나요?",
    "한국어로 글을 생성할 수 있나요?",
]

# 추론 실행
for text in input_texts:
    # 입력 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # 모델 추론
    outputs = model.generate(**inputs, max_length=50)

    # 결과 디코딩 및 출력
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"입력: {text}\n출력: {result}\n")

# 작업 종료 후 GPU 메모리 정리
# del model
# torch.cuda.empty_cache()
