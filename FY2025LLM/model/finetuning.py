""" Llama 3.2 1B 파인튜닝
# python FY2025LLM/model/finetuning.py
# 환경
@OS Windows
@Python 3.12.1
@NVIDIA-SMI 560.94 / Driver Version: 560.94 / CUDA Version: 12.6
@torch 2.5.1+cu121
@trl 0.18.1
@transformers 4.52.4
@dataset 3.6.0
"""
# from torch.testing._internal.common_nn import padding_mode PyTorch의 nn.Conv2d 등에서 쓰이는 옵션 중 하나로, LLM 에서는 불필요하다고 함.
from transformers import default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from langdetect import detect
# LlamaTokenizerFast를 명시적으로 import 한다. 'tokenizer.model' 파일이 올바르게 저장되도록 보장하기 위함이다.
from transformers import LlamaTokenizerFast
from datasets import load_dataset
from transformers import EarlyStoppingCallback # 학습 조기종료
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from trl import SFTTrainer, SFTConfig
import json
import os
from peft import get_peft_model, TaskType, LoraConfig, PeftModel
from config import conf
from pathlib import Path

base_dir = Path(conf.PROJECT_ROOT_DIRECTORY)
MODEL_PATH   = str( base_dir / "models" / "llama3.2-1B-hf" )
# DATASET_PATH = str( base_dir / "data" / "converted" / "CarrotAI" / "cleaned_llama3_dataset.jsonl" )
# SAVE_PATH    = str( base_dir / "models" / "llama3.2-1B-hf" / "finetuned" / "model_v3" )
DATASET_PATH = str( base_dir / "data" / "converted" / "merged" / "train_total.jsonl" )
SAVE_PATH    = str( base_dir / "models" / "llama3.2-1B-hf" / "finetuned" / "model_v4" )
os.makedirs(SAVE_PATH, exist_ok=True)

# GPU 사용 여부 확인, 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 성능 확인
print(torch.cuda.get_device_name(0))
print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
print(f"Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

""" Step 1. 토크나이저 셋팅 """
# tokenizer & model 로드 : 로컬 디렉토리에서 불러오기
# 토크나이저 로드: AutoTokenizer 대신 LlamaTokenizerFast를 사용
tokenizer = LlamaTokenizerFast.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True,
    legacy = False # legacy=True는 기존 방식,
)
tokenizer.pad_token = tokenizer.eos_token # 패딩 설정
tokenizer.padding_side = "right" # 오른쪽으로 패딩을 붙임, Llama, GPT 계열

# special_tokens_dict = {
#     "pad_token": tokenizer.eos_token  # 또는 원하는 토큰
# }
# tokenizer.add_special_tokens(special_tokens_dict)


# 공식 LLaMA 3.2용 chat template 수동 삽입
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
)
# print("추가된 special token ID들:", tokenizer.added_tokens_encoder.keys())
# print("내용:", tokenizer.added_tokens_encoder)
# set CUDA_LAUNCH_BLOCKING=1 을 터미널에서 설정하게 디버깅

""" ================================================ 데이터 전처리 =============================================="""
# 1. 데이터셋 로드 (JSONL)
dataset = load_dataset("json", data_files=DATASET_PATH)
# 2. 토큰 길이를 기준으로 너무 큰 데이터는 필터링
MAX_LENGTH = 1024
def check_token_length(example):
    formatted_text = tokenizer.apply_chat_template( # 대화 내용을 템플릿에 적용하여 하나의 문자열로 만든다.
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    return len(tokenizer.encode(formatted_text)) <= MAX_LENGTH # 토크나이징 후 길이를 계산하여 반환한다. (MAX_LENGTH보다 긴 데이터를 제거)
print(f"Original dataset size: {len(dataset['train'])}")
filtered_dataset = dataset.filter(check_token_length)
print(f"Filtered dataset size (<= {MAX_LENGTH} tokens): {len(filtered_dataset['train'])}")
# 3. 소스 구동을 위해 (실제 파인튜닝이 아닌, 파인튜닝 소스 체크 용도) 데이터의 양을 랜덤으로 축소시킨다.
SAMPLE_SIZE = 2000
# 필터링된 데이터셋을 섞은 후, 앞에서부터 SAMPLE_SIZE 만큼 선택, 만약 필터링된 데이터가 SAMPLE_SIZE보다 적으면, 가능한 모든 데이터를 사용.
num_available_samples = len(filtered_dataset["train"])
if num_available_samples < SAMPLE_SIZE:
    print(f"Warning: Available samples ({num_available_samples}) is less than the requested sample size ({SAMPLE_SIZE}). Using all available samples.")
    sample_size_to_use = num_available_samples
else:
    sample_size_to_use = SAMPLE_SIZE

final_dataset_for_split = filtered_dataset["train"].shuffle(seed=42).select(range(sample_size_to_use))
print(f"Sampled dataset size for training/evaluation: {len(final_dataset_for_split)}")


# 샘플링된 데이터셋을 학습/검증용으로 분할
split_dataset = final_dataset_for_split.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# 데이터 전처리
# SFTTrainer가 내부적으로 데이터를 처리하도록 `formatting_func`를 정의, (trl 라이브러리에서 권장하는 방법)
def formatting_prompts_func(example):
    # example['messages']는 [{'role': 'user', ...}, {'role': 'assistant', ...}] 형태의 리스트
    # apply_chat_template을 사용하면 SFTTrainer가 내부적으로 prompt 부분의 loss를 무시하도록 처리한다.
    # return [tokenizer.apply_chat_template(conversation, tokenize=False) for conversation in example["messages"]]
    prompt = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return [prompt]

# 모델 로드
# bf16 지원 여부를 확인하여 학습 안정성을 높인다.
is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
if not is_bf16_supported:
    print("\nWarning: Your GPU does not support bf16. Falling back to fp16, which can be unstable.\n")
else:
    print(f"is_bf16_supported : {is_bf16_supported}")

""" Trainer 설정 """
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=2, # 2
    # max_steps=5, # 성능 생각하지 않고 일단 로컬PC에서 가동해보려는 용도, max_steps 를 쓰면 num_train_epochs 보다 우선 적용된다.
    per_device_train_batch_size=1,
    # 작은 배치 사이즈를 보완하기 위해 그래디언트 축적(accumulation)을 사용한다.
    # 실질적인 배치 사이즈는 batch_size * accumulation_steps = 1 * 4 = 4가 된다.
    gradient_accumulation_steps=3, # 키울 수록 메모리가 절약된다.
    # bf16은 더 넓은 동적 범위를 가져 fp16보다 수치적으로 훨씬 안정적이라고 한다.
    bf16=is_bf16_supported, # (Ampere 아키텍처 GPU(예: RTX 30xx, A100) 이상에서 지원)
    fp16=not is_bf16_supported, # bf16을 지원하지 않는 경우에만 fp16을 사용한다. (오류 발생 가능성 있음)
    learning_rate=1e-5,
    logging_steps=100,
    save_steps=1000,
    eval_strategy="steps", # 학습 도중 주기적으로 평가(evaluation)를 수행하도록 설정
    eval_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # optim="paged_adamw_32bit", # For linux
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=False,
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    # quantization_config=quant_config, Linux 용 옵션
    # device_map="auto" # vRAM 분산 및 모델 분할 로딩에 유리하다고 한다. TODO 공부하기
    # accelerate를 사용해서 자동으로 GPU와 CPU를 나눠서 로딩 ... Trainer와 충돌하여 주석
    # 모델을 32로 부르고, Trainer 에서 16으로 변환하려고 하면 CUDA Memory out 이 난다.
    torch_dtype=torch.bfloat16 if is_bf16_supported else torch.float16,
    attn_implementation="sdpa" # Flash Attention 은 Linux 환경에서만 사용 가능
)

tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # ① 먼저 special token 추가
model.resize_token_embeddings(len(tokenizer))                     # ② 그 후 embedding 사이즈를 조정


# LoRA 적용
model = get_peft_model(model, peft_params) # peft 로 하자 6시간30분 -> 1시간 30분 정도로 줄어들었다.
model.print_trainable_parameters()  # 확인용


# 모델의 첫 번째 파라미터 dtype 출력 (float 확인용도) ... Loss nan 이 발생하여 확인
print(next(model.parameters()).dtype) # 기본: torch.float32
model.to(device)  # 직접 GPU로 이동



token_lengths = []  # 분석용

# 학습 전 확인
print(f"Model dtype: {model.dtype}")
print(f"Model device: {next(model.parameters()).device}")

# 학습 시작
print("Start training...")
# # SFTTrainer 정의
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 전처리되지 않은 원본 데이터셋을 전달
    eval_dataset=eval_dataset,  # 전처리되지 않은 원본 데이터셋을 전달
    formatting_func=formatting_prompts_func,  # 위에서 정의한 포매팅 함수 지정
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # 검증 성능이 개선되지 않는 epoch가 2번 연속 발생하면 학습을 중단.
)

"""  Full FineTuning END """



# 학습
trainer.train()




# 모델 저장
print("===== 내부 backend 확인 =====")#
print("Tokenizer class:", tokenizer.__class__)
print("Has backend_tokenizer:", hasattr(tokenizer, "backend_tokenizer"))
print("Contains tokenizer.model?", os.path.exists(os.path.join(SAVE_PATH, "tokenizer.model")))
print("===== 모델 저장 직전 토크나이저와 모델 정보 확인 =====")
print("tokenizer vocab size:", tokenizer.vocab_size) # 모델 학습 전 원래 vocab 크기
print("model embedding size:", model.get_input_embeddings().num_embeddings) # 스페셜 토큰 추가 후 학습에 쓰였을때의 크기
print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
print("tokenizer.special_tokens_map:", tokenizer.special_tokens_map)

# trainer.model.save_pretrained(SAVE_PATH)
# trainer.processing_class .save_pretrained(SAVE_PATH)
# 파인튜닝 결과 저장
trainer.save_model(SAVE_PATH) # 모델 저장
tokenizer.save_pretrained(SAVE_PATH) # 토크나이저 저장
print(f"모델 및 토크나이저 저장 완료: {SAVE_PATH}")


# 저장 후 확인
print("\n===== 저장 후 디렉토리 파일 목록 =====")
print(os.listdir(SAVE_PATH))
print(f"tokenizer.model 파일 존재 여부: {os.path.exists(os.path.join(SAVE_PATH, 'tokenizer.model'))}")
#
"""
finetunded 경로에 저장된 데이터를 확인해보니 tokenizer.model 이 저장되지 않고 있었다. A
uto가 아닌 LlamaTokenizer 를 써야 저장되는 것 같다고..?? 하는데..
>> tokenizer.model은 LlamaTokenizer (slow) 전용으로,  Fast 에서는 필요 없다고 한다.

tokenizer.json             : Fast tokenizer의 핵심 파일 (tokenizer.model의 역할을 대신함)
tokenizer_config.json      : 토크나이저 클래스, padding 등 설정 정보
special_tokens_map.json    : <pad>, <eos> 등 스페셜 토큰 ID 매핑
adapter_config.json        : PEFT(LoRA)로 학습된 가중치
adapter_model.safetensors  : PEFT(LoRA)로 학습된 가중치
training_args.bin          : 학습 시 사용한 TrainingArguments 저장
chat_template.jinja        : 직접 삽입한 chat template 
checkpoint-5/              : 학습 중 저장된 체크포인트
runs/                      : TensorBoard 로그 폴더 (옵션에 따라 생김)

"""

"""

"""
# $env:PYTHONPATH = (Get-Location).Path
#  PYTHONPATH가 제대로 설정되었는지 확인
# echo $env:PYTHONPATH
# python model/finetuning.py