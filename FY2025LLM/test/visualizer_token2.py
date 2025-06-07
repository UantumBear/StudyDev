""" Llama 3.2 3B 파인튜닝을 위한, 학습용으로 다운받은 데이터 시각화

$ python FY2025LLM/test/visualizer_token2.py
"""


from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import json

# 경로 설정
jsonl_path = "FY2025LLM/data/converted/CarrotAI/ko_instruction_to_llama3_format.jsonl"
model_path = "FY2025LLM/models/llama3.2-1B-hf"

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
)

# 토큰 길이 수집
user_lengths = []
assistant_lengths = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = data["messages"]

        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        assistant_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")

        user_len = len(tokenizer(user_text)["input_ids"]) if user_text else 0
        assistant_len = len(tokenizer(assistant_text)["input_ids"]) if assistant_text else 0

        user_lengths.append(user_len)
        assistant_lengths.append(assistant_len)

# 시각화
plt.figure(figsize=(10, 6))
plt.hist(user_lengths, bins=50, alpha=0.6, label="User", color="skyblue", edgecolor="black")
plt.hist(assistant_lengths, bins=50, alpha=0.6, label="Assistant", color="salmon", edgecolor="black")
plt.title("Token Length Distribution by Role")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("FY2025LLM/test/data/visual/rolewise_token_hist.png")
plt.show()
