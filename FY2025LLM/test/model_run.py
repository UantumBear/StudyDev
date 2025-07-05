# model_run.py

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast
# PreTrainedTokenizer 는 추상클래스로 이것을 가져다 쓰지 않음.
import torch
from pathlib import Path
from peft import PeftModel
from config.conf import PROJECT_ROOT_DIRECTORY

# 한번 테스트해봅니다. 헤헤 파일이 감시되려나?


# === 경로 설정 ===
# MODEL_DIR = f"{PROJECT_ROOT_DIRECTORY}/models/llama3.2-1B-hf/finetuned/model_v1"

# === 경로 설정 ===
base_model_path = str(Path(PROJECT_ROOT_DIRECTORY) / "models" / "llama3.2-1B-hf")
adapter_path = str(Path(PROJECT_ROOT_DIRECTORY) / "models" / "llama3.2-1B-hf" / "finetuned" / "model_v1")

# === 디바이스 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 토크나이저 로드 ===
tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)


# === base 모델 로드 ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # vLLM 쓰는 게 아니라면 괜찮습니다
)

# === LoRA adapter 붙이기 ===
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()  # 평가 모드로 전환
# model.to(device) ← device_map="auto" 덕분에 보통 생략 가능. 명시하고 싶으면 OK



#
# !! tokenizer vocal 과 model embedding 의 size 가 동일해야 한다.
print("tokenizer vocab size:", tokenizer.vocab_size)
print("model embedding size:", model.get_input_embeddings().num_embeddings)
assert model.get_input_embeddings().num_embeddings == tokenizer.vocab_size



# === 대화 시작 ===
print("챗봇과 대화를 시작하세요. 'exit' 입력 시 종료됩니다.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    if not user_input.strip():
        print("입력이 비어 있습니다. 다시 입력해주세요.")
        continue

    messages = [
        {"role": "system", "content": "당신은 친절하고 똑똑한 한국어 인공지능 비서입니다. 질문에 친절하고 명확하게 응답하세요."},
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_special_tokens=False

    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    print("input_ids:", input_ids)
    print("attention_mask:", attention_mask)
    print("디코딩된 prompt:\n", tokenizer.decode(input_ids[0], skip_special_tokens=False))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Assistant: {response.strip()}")


# export PYTHONPATH=/home/devbear/dev_projects/StudyDev/FY2025LLM
# python FY2025LLM/test/model_run.py
