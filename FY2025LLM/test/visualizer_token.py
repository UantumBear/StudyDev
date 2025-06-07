""" Llama 3.2 3B 파인튜닝을 위한, 학습용으로 다운받은 데이터 시각화

$ python FY2025LLM/test/visualizer_token.py
"""

import os
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import json

# 데이터 경로
jsonl_path = "FY2025LLM/data/converted/CarrotAI/ko_instruction_to_llama3_format.jsonl"
model_path = "FY2025LLM/models/llama3.2-1B-hf"
output_dir = "FY2025LLM/test/data/visual"
output_img_path = os.path.join(output_dir, "token_length_histogram.png")

# 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

# chat_template 수동 삽입
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
)

# 토큰 길이 측정
lengths = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = data["messages"]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(chat_text)["input_ids"]
        lengths.append(len(tokens))

# 히스토그램 시각화 및 저장
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title("Token Length Distribution (After chat_template)")
plt.xlabel("Number of tokens")
plt.ylabel("Number of samples")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_img_path)  # 이미지 저장
print(f" 히스토그램 저장 완료: {output_img_path}")