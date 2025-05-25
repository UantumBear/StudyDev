""" Llama 3.2 3B 파인튜닝

$ python FY2025LLM/test/finetuning.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import json

# 경로 설정
MODEL_PATH = "FY2025LLM/models/llama3.2-3B-hf"
DATASET_PATH = "FY2025LLM/data/converted/CarrotAI/ko_instruction_to_llama3_format.jsonl"

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer & model 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# 공식 LLaMA 3.2용 chat template 수동 삽입
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
)

tokenizer.pad_token = tokenizer.eos_token  # 패딩 설정
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

# 데이터셋 로드 (JSONL)
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 토크나이즈 함수
def tokenize(example):
    result = tokenizer.apply_chat_template(example["messages"], truncation=True, max_length=2048)
    return tokenizer(result, return_tensors="pt", padding="max_length", max_length=2048, truncation=True)

tokenized_dataset = dataset.map(tokenize, remove_columns=["messages"], batched=True)

# 데이터 정렬용 Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 학습 설정
training_args = TrainingArguments(
    output_dir="FY2025LLM/models/llama3.2-3B-hf/checkpoints/llama3-ko",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    logging_dir="./logs",
)

# Trainer 설정 및 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
