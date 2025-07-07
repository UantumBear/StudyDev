# llama3_run.py
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from data.DevBear import system_prompt as pmpt

# MODEL_PATH = "models/llama3.2-1B-hf"  # 원본 모델 경로
MODEL_PATH = ("models/meta-llama/Llama-3.2-1B-Instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"[Device] {device} | bf16={is_bf16}")

# 1. Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    legacy=False  # LLaMA3 tokenizer 최신방식 권장
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 공식 ChatML 템플릿
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
    "{% endif %}{% endfor %}"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# 2. 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if is_bf16 else torch.float16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 3. Generation config 설정
generation_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id
)

# 4. 대화 루프
print("🌟 LLaMA3 원본 모델 대화 시작 (종료하려면 'exit') 🌟")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [
        {
            "role": "system",
            "content": (
                pmpt.PROMPT_V1
            )
        },
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )

    output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Assistant: {output_text.strip()}")

### 파인튜닝 되지 않은 모델은 기본 모델로 챗봇에는 적합하지 않은 모습
# python model/run.py