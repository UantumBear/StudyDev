""" Llama 3.2 3B 파인튜닝

$ python FY2025LLM/test/finetuning.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
import os

# 경로 설정
MODEL_PATH = "FY2025LLM/models/llama3.2-1B-hf"
DATASET_PATH = "FY2025LLM/data/converted/CarrotAI/ko_instruction_to_llama3_format.jsonl"

# GPU 사용 여부 확인, 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 성능 확인
print(torch.cuda.get_device_name(0))
print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
print(f"Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

# tokenizer & model 로드 : 로컬 디렉토리에서 불러오기
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token # 패딩 설정
# 공식 LLaMA 3.2용 chat template 수동 삽입
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16
    # device_map="auto" # vRAM 분산 및 모델 분할 로딩에 유리하다고 한다. TODO 공부하기
    # accelerate를 사용해서 자동으로 GPU와 CPU를 나눠서 로딩 ... Trainer와 충돌하여 주석
)
# 모델의 첫 번째 파라미터 dtype 출력 (float 확인용도) ... Loss nan 이 발생하여 확인
print(next(model.parameters()).dtype) # 기본: torch.float32
model.to(device)  # 직접 GPU로 이동

# 데이터셋 로드 (JSONL)
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
# 2. 메시지 토큰 수가 512 미만인 샘플만 남김...자꾸 Loss nan     이 발생하여 적용해봄.
def filter_too_long(example):
    chat = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return len(tokenizer(chat)["input_ids"]) < 256
# 3. 필터링 적용
dataset = dataset.filter(filter_too_long)

# Tokenize 함수
def tokenize(batch):
    """
    LaMA3.2 모델은 Supervised Fine-Tuning (SFT) 시 모델이 assistant 응답만 학습하도록 해야 한다고 한다.
    labels 설정 시 user의 부분은 -100으로 마스킹 () 하기
    """

    input_ids_list, attention_mask_list, labels_list = [], [], []

    for messages in batch["messages"]:
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)

        # 전체 토큰화
        inputs = tokenizer(chat_text, padding="max_length", max_length=256, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.copy()

        # user 발화까지는 -100으로 마스킹
        try:
            user_end = chat_text.index("<|start_header_id|>assistant<|end_header_id|>")
            user_tokenized = tokenizer(chat_text[:user_end], padding="max_length", max_length=256, truncation=True)
            mask_len = len(user_tokenized["input_ids"])
            labels[:mask_len] = [-100] * mask_len
        except ValueError:
            # assistant 부분이 없을 경우, 전체 마스킹
            labels = [-100] * len(labels)

        # 리스트에 추가
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["messages"])

# DataLoader
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, collate_fn=collator) # batch 4 로는 GPU가 터진다.

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6) # 1e-5 로 하자 계속해서 Loss: nan 이 발생했다.


# scaler = GradScaler()

# 메모리 누수 방지
# torch.cuda.empty_cache()

# 학습 전 확인
print(f"Model dtype: {model.dtype}")
print(f"Model device: {next(model.parameters()).device}")

# 학습 루프
model.train()


for epoch in range(1):
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        # forward
        outputs = model(**batch)
        loss = outputs.loss

        # NaN/Inf 방지: loss 검증
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Step {step}] Loss is NaN/Inf. Skipping step.")
            continue

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}")
            print(" Sample input_ids:", batch["input_ids"][0][:30])
            print(" Decoded input:", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

# 모델 저장
save_path = "FY2025LLM/models/llama3.2-1B-hf/finetuned/model_v1"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)