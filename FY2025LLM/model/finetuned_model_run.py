"""
@경로 model/finetuned_model_run.py
@목적 models/llama3.2-1B-hf/finetuned 하위의 llama3 Base 모델을 챗봇형 데이터로 파인튜닝한 모델을 실행하여 테스트 하는 역할
"""


from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast
from transformers import GenerationConfig
# PreTrainedTokenizer 는 추상클래스로 이것을 가져다 쓰지 않음.
import torch
from pathlib import Path
from peft import PeftModel
from config import conf
from pathlib import Path


# --------------------------------------------------
# 1) 경로 & 디바이스 설정
# --------------------------------------------------
base_dir = Path(conf.PROJECT_ROOT_DIRECTORY)
BASE_MODEL_ORIGINAL_PATH  = str( base_dir / "models" / "llama3.2-1B-hf" ) # 원본 모델 경로
# ADAPTER_PATH    = str( base_dir / "models" / "llama3.2-1B-hf" / "finetuned" / "model_v2" ) # 어댑터 패스
ADAPTER_PATH    = str( base_dir / "models" / "llama3.2-1B-hf" / "finetuned" / "model_v4" ) # 어댑터 패스
"""
PEFT (Parameter-Efficient Fine-Tuning) 기법 중 하나인 LoRA (Low-Rank Adaptaion) 을 사용하여 모델을 파인튜닝 한 경우,
파인튜닝 된 모델을 로드하는 방식은 아래와 같다.

1. 원본 모델 (Pre-trained) 을 로드한다.
LoRA는 원본 모델의 모든 가중치를 수정하는 것이 아니라, 원본 모델의 특정 레이어 옆에 작은 "adapter" 행렬을 추가하여 학습하는 것이기에,
추론 시에도 원본 모델의 대부분의 가중치가 필요하다.
2. 로드된 원본 모델에 파인튜닝된 LoRA 어댑터 가중치를 연결(Attach) 한다.
peft 라이브러리의 PeftModel.from_pretrained(base_model, adapter_path) 함수가 그 역할을 한다.
원본 모델에 adapter_path 에 저장된 LoRA 가중치를 통합하여, 하나의 PeftModel 객체를 만드는 것이다.

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
    print(f"Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")
# BF16 지원 여부 확인
is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"[Device] {device}  |  bf16={is_bf16}")


# --------------------------------------------------
# 2) 토크나이저 로드 (가장 먼저, adapter 경로에서) (파인튜닝된 토크나이저 설정을 포함)
# --------------------------------------------------
tokenizer = LlamaTokenizerFast.from_pretrained(
    ADAPTER_PATH, # finetuned/model_v2 경로에서 토크나이저 로드
    use_fast=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token            # pad-id = eos-id
tokenizer.padding_side = "right"                     # Llama 계열은 우측 padding
# print("실제 모델 임베딩 크기:", model.get_input_embeddings().num_embeddings)
# tokenizer_vocab_size = model.get_input_embeddings().num_embeddings
tokenizer.chat_template = ( # 파인튜닝 시와 동일하게 설정 (Llama 3 공식 템플릿)
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
    "<|start_header_id|>assistant<|end_header_id|>\n"  # generation 시작 지점을 명시
)

# --------------------------------------------------
# 3) base 모델 로드 → 임베딩 크기 맞추기
# --------------------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ORIGINAL_PATH, # <-- 원본 베이스 모델 경로
    torch_dtype=torch.bfloat16 if is_bf16 else torch.float16, # 학습 시와 동일하게
    device_map="auto" # vRAM 분산 및 모델 분할 로딩 (GPU가 하나면 하나에만 올라감)
)
base_model.resize_token_embeddings(len(tokenizer))   # 모델의 임베딩 크기를 토크나이저 어휘 크기에 맞춤


# --------------------------------------------------
# 4) LoRA 어댑터 붙이기
# --------------------------------------------------
# base_model에 파인튜닝된 어댑터를 로드하여 최종 모델을 생성
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) # <-- 파인튜닝된 어댑터 경로 사용
model.eval() # 평가 모드로 전환
print("!어댑터가 제대로 로드되었는지 확인! Loaded adapters:", model.peft_config.keys())

generation_config = GenerationConfig(
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)


# !! tokenizer vocab 과 model embedding 의 size 가 동일해야 한다.
print(f"vocab(tokenizer)={tokenizer.vocab_size},  embedding(model)={model.get_input_embeddings().num_embeddings}")
actual_vocab_size = len(tokenizer)
model_vocab_size = model.get_input_embeddings().num_embeddings

print(f"Tokenizer actual vocab (special token 포함한 실제 토큰 개수) : {actual_vocab_size}")
print(f"Model embedding size:   {model_vocab_size}")

assert actual_vocab_size == model_vocab_size, "❌ tokenizer와 model의 vocab 수가 다릅니다!"

# --------------------------------------------------
# 5) 대화 루프
# --------------------------------------------------
print("\n챗봇과 대화를 시작하세요. 'exit' 입력 시 종료됩니다.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        print("빈 입력입니다. 다시 입력해 주세요.")
        continue

    messages = [
        {"role": "system", "content": "당신은 한국어로만 정중하고 명확하게 응답하는 AI 챗봇입니다. 사용자의 의도를 잘 파악하여 친절하게 대답하세요."},
        {"role": "user",   "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # ← 템플릿이 이 플래그를 인식해야 함
        return_tensors="pt"
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # EOT 토큰(id) 안전하게 추출
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id] + ([eot_id] if eot_id is not None else [])

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config  # 직접 전달!
    )

    answer = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"Assistant: {answer}")

# export PYTHONPATH=/home/devbear/dev_projects/StudyDev/FY2025LLM
# python model/run.py
