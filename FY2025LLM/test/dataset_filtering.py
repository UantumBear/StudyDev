# python FY2025LLM/test/dataset_filtering.py

import json
from transformers import AutoTokenizer

# 사용자 환경 경로로 변경
MODEL_PATH = "FY2025LLM/models/llama3.2-1B-hf"
INPUT_FILE = "FY2025LLM/data/converted/CarrotAI/ko_instruction_to_llama3_format.jsonl" # 원본은 llama 형태로
OUTPUT_FILE = "FY2025LLM/data/converted/CarrotAI/cleaned_llama3_dataset.jsonl" # 토큰길이, 형식 등 전처리
REPORT_FILE = "FY2025LLM/data/converted/CarrotAI/llama3_dataset_issues_report.json" # 리포트

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
)

# 검사 및 필터링
cleaned_data = []
invalid_samples = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            sample = json.loads(line)
            messages = sample.get("messages", [])

            roles = [msg.get("role") for msg in messages]
            if "user" not in roles or "assistant" not in roles:
                invalid_samples.append((idx, "missing_role"))
                continue

            assistant_msgs = [msg.get("content", "").strip() for msg in messages if msg["role"] == "assistant"]
            if not assistant_msgs or any(msg == "" for msg in assistant_msgs):
                invalid_samples.append((idx, "empty_assistant"))
                continue

            chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
            input_ids = tokenizer(chat_text)["input_ids"]
            if not (500 <= len(input_ids) <= 4096):
                invalid_samples.append((idx, f"bad_length_{len(input_ids)}"))
                continue

            cleaned_data.append(sample)

        except Exception as e:
            invalid_samples.append((idx, f"exception_{str(e)}"))

# 결과 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in cleaned_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(invalid_samples, f, ensure_ascii=False, indent=2)

print(f"✅ 정상 샘플: {len(cleaned_data)}개")
print(f"⚠️ 문제 샘플: {len(invalid_samples)}개 → {REPORT_FILE}에서 확인 가능")
