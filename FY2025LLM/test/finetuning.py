""" Llama 3.2 1B 파인튜닝

# python FY2025LLM/test/finetuning.py
"""
from transformers import default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import LlamaTokenizerFast
from datasets import load_dataset
from transformers import EarlyStoppingCallback # 학습 조기종료
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from trl import SFTTrainer, SFTConfig
import json
import os

# 경로 설정
MODEL_PATH = "FY2025LLM/models/llama3.2-1B-hf"
DATASET_PATH = "FY2025LLM/data/converted/CarrotAI/cleaned_llama3_dataset.jsonl"
SAVE_PATH = "FY2025LLM/models/llama3.2-1B-hf/finetuned/model_v1"
os.makedirs(SAVE_PATH, exist_ok=True)

# GPU 사용 여부 확인, 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 성능 확인
print(torch.cuda.get_device_name(0))
print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
print(f"Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

# tokenizer & model 로드 : 로컬 디렉토리에서 불러오기
tokenizer = LlamaTokenizerFast.from_pretrained(
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


# set CUDA_LAUNCH_BLOCKING=1 을 터미널에서 설정하게 디버깅


# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16
    # device_map="auto" # vRAM 분산 및 모델 분할 로딩에 유리하다고 한다. TODO 공부하기
    # accelerate를 사용해서 자동으로 GPU와 CPU를 나눠서 로딩 ... Trainer와 충돌하여 주석
    # 모델을 32로 부르고, Trainer 에서 16으로 변환하려고 하면 CUDA Memory out 이 난다.
)
model.resize_token_embeddings(len(tokenizer))
# 모델의 첫 번째 파라미터 dtype 출력 (float 확인용도) ... Loss nan 이 발생하여 확인
print(next(model.parameters()).dtype) # 기본: torch.float32
model.to(device)  # 직접 GPU로 이동

# 데이터셋 로드 (JSONL)
dataset = load_dataset("json", data_files=DATASET_PATH)
# 학습:검증 = 80:20 분할 (shuffle 포함)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]



# # 첫 번째 샘플 출력
# print("dataset[0]:")
# print(dataset[0])
# chat_text = tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=False)
# print("chat_text:")
# print(chat_text)



MAX_LENGTH = 2048
MIN_LENGTH = 500
token_lengths = []  # 분석용


# Tokenize 함수
def tokenize(batch):
    """
    LaMA3.2 모델은 Supervised Fine-Tuning (SFT) 시 모델이 assistant 응답만 학습하도록 해야 한다고 한다.
    labels 설정 시 user의 부분은 -100으로 마스킹 () 하기
    """
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"

    for messages in batch["messages"]:
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)

        try:
            # 1. assistant 블록 위치 찾기
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
            if assistant_marker not in chat_text:
                continue
            # 문맥 추출
            start = chat_text.index(assistant_marker)
            context = chat_text[max(0, start - 800):]

            # 토큰화
            enc = tokenizer(context, padding="max_length", max_length=MAX_LENGTH, truncation=True)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels = input_ids.copy()

            # 마스킹 처리
            prefix_len = len(tokenizer(context[:context.index(assistant_marker)])["input_ids"])
            labels[:prefix_len] = [-100] * prefix_len

            if all(l == -100 for l in labels):
                continue

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        except Exception:
            continue

        return results if results["input_ids"] else None

# 전처리 적용
tokenized_train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["messages"],
    desc="Tokenizing train dataset"
)
tokenized_eval_dataset = eval_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["messages"],
    desc="Tokenizing eval dataset"
)




## 직접 torch 로 train 을 하면 계속 loss nan 이 발생해서 일단 삭제..
# DataLoader
# collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False
# )
# DataCollatorForLanguageModeling은 보통 MLM(Masked Language Modeling)용이며, Causal LM에는 맞지 않을 수 있다고 한다.
# Dataloader
# collator = default_data_collator
# dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, collate_fn=collator) # batch 4 로는 GPU가 터진다.

# Optimizer & AMP
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6) # 1e-5 로 하자 계속해서 Loss: nan 이 발생했다.
# scaler = GradScaler(device='cuda')  # <-- 디바이스 명시!

# 메모리 누수 방지
# torch.cuda.empty_cache()

# 학습 전 확인
print(f"Model dtype: {model.dtype}")
print(f"Model device: {next(model.parameters()).device}")

# 학습 시작
print("Start training...")



# Trainer 학습 파라미터
# training_args = TrainingArguments(
#     output_dir=SAVE_PATH,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=1,
#     num_train_epochs=1,
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=1,
#     learning_rate=5e-6,
#     # bf16=False, # bfloat16 → False
#     # fp16=True,  # AMP 적용 (자동)
#     report_to="none"
# )
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps", # 학습 도중 주기적으로 평가(evaluation)를 수행하도록 설정
    eval_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    learning_rate=5e-6,
    report_to="none"
)
# SFTTrainer 정의
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # 검증 성능이 개선되지 않는 epoch가 2번 연속 발생하면 학습을 중단.
)

# 학습
trainer.train()




# 모델 저장
print("===== 내부 backend 확인 =====")#
print("Tokenizer class:", tokenizer.__class__)
print("Has backend_tokenizer:", hasattr(tokenizer, "backend_tokenizer"))
print("Contains tokenizer.model?", os.path.exists(os.path.join(SAVE_PATH, "tokenizer.model")))
print("===== 모델 저장 직전 토크나이저와 모델 정보 확인 =====")
print("📌 tokenizer vocab size:", tokenizer.vocab_size)
print("📌 model embedding size:", model.get_input_embeddings().num_embeddings)
print("📌 tokenizer.pad_token_id:", tokenizer.pad_token_id)
print("📌 tokenizer.eos_token_id:", tokenizer.eos_token_id)
print("📌 tokenizer.special_tokens_map:", tokenizer.special_tokens_map)

trainer.model.save_pretrained(SAVE_PATH)
trainer.processing_class.save_pretrained(SAVE_PATH)
print(f"모델 저장 완료: {SAVE_PATH}")
tokenizer.save_pretrained(SAVE_PATH)
print(f"토크나이저 저장 완료: {SAVE_PATH}")

"""
데이터를 확인해보니 tokenizer.model 이 저장되지 않고 있었다. Auto가 아닌 LlamaTokenizer 를 써야 저장되는 것 같다고..
"""

# python FY2025LLM/test/finetuning.py