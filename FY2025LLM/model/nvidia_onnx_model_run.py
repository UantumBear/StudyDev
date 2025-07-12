import onnxruntime as ort
import numpy as np
import os
from transformers import AutoTokenizer
from scipy.special import softmax


# --- 설정 ---
# 1. ONNX 모델 경로
ONNX_MODEL_DIR =  f"models/nvidia/meta-llama-3.2-3b-onnx-int4-rtx_v1.0/llama32_onnx_int4_genai_dml" # .onnx 파일이 있는 디렉토리
ONNX_MODEL_PATH = os.path.join(ONNX_MODEL_DIR, "model.onnx") # model.onnx 파일명 확인

import os
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time


# 3. 추론 설정
MAX_NEW_TOKENS = 200  # 최대 생성할 토큰 수
TEMPERATURE = 0.7  # 샘플링 온도 (높을수록 창의적)
TOP_P = 0.9  # Top-p 샘플링 (높을수록 다양성 증가)

# 4. ONNX Runtime Execution Provider 설정
# TensorRT EP를 최우선으로 사용
# TensorRT Engine Cache Path: 처음 실행 시 엔진 빌드 시간이 오래 걸릴 수 있으므로 캐싱 권장
TENSORRT_CACHE_PATH = os.path.join(ONNX_MODEL_DIR, "trt_cache")
os.makedirs(TENSORRT_CACHE_PATH, exist_ok=True)

PROVIDERS = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": False,  # INT4 모델이므로 FP16 변환 필요 없음. (단, 모델이 FP16으로 내보내진 경우 True)
        "trt_int8_enable": True,  # INT4 모델이므로 INT8 모드를 사용하도록 명시 (TensorRT 8.x 이상에서 INT4 처리)
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": TENSORRT_CACHE_PATH
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

# --- 토크나이저 로드 ---
print(f"토크나이저 로드 중: {ONNX_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(
    ONNX_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Llama 3는 pad_token이 없음


# Llama 3 Chat Template (Hugging Face 형식)
# 이 템플릿은 대화의 시작/끝, 역할 등을 정의
# https://huggingface.co/docs/transformers/main/chat_templets#llama-3
def apply_llama_chat_template(messages, add_generation_prompt=True):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # True로 하면 토큰 ID 리스트 반환
        add_generation_prompt=add_generation_prompt
    )


# --- ONNX 세션 생성 ---
print(f"ONNX 모델 로드 중: {ONNX_MODEL_PATH}")
print(f"사용 가능한 EP: {ort.get_available_providers()}")
print(f"사용할 EP: {PROVIDERS}")

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=PROVIDERS
)

# 입력/출력 이름 확인
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]

# KV 캐시 관련 이름 추출 (model.onnx 파일에 따라 이름이 다를 수 있으니 확인 필요)
# Llama 모델은 보통 past_key_values.0, past_key_values.1 ... present.0, present.1... 형태
past_kv_names = sorted([name for name in input_names if "past_key_values" in name])
present_kv_names = sorted([name for name in output_names if "present" in name])

print(f"ONNX 입력 이름: {input_names}")
print(f"ONNX 출력 이름: {output_names}")
print(f"Past KV 이름: {past_kv_names}")
print(f"Present KV 이름: {present_kv_names}")

# 모델 구성에서 head_dim, num_heads, num_layers 가져오기 (KV 캐시 shape용)
# NGC 모델은 .onnx 파일만 있으므로 직접 Config를 로드하기 어렵습니다.
# Llama 3 3B 모델의 일반적인 값:
# hidden_size = 3200
# num_hidden_layers = 26
# num_attention_heads = 32
# head_dim = hidden_size // num_attention_heads = 3200 // 32 = 100

# 만약 정확한 값을 모른다면, 더미 KV 캐시의 shape을 통해 유추하거나,
# Hugging Face의 Llama 3-3B 모델 Config를 참고하세요.
# 여기서는 3B 모델의 일반적인 값으로 가정합니다.
# 이 값들은 1B나 8B와 다릅니다! 정확히 아셔야 합니다.
DUMMY_BATCH_SIZE = 1  # 배치 사이즈는 1로 고정
DUMMY_NUM_HEADS = 8  # ONNX Runtime 오류 메시지에서 'Expected: 8' 확인
DUMMY_HEAD_DIM = 128  # ONNX Runtime 오류 메시지에서 'Expected: 128' 확인
DUMMY_NUM_LAYERS = 28 # ONNX 입력/출력 이름을 통해 실제 레이어 수 28을 확인!!


# --- 샘플링 함수 (그리디 디코딩 대신) ---
def sample_next_token(logits, temperature, top_p):
    logits = logits[0, -1, :]  # 현재 마지막 토큰의 로짓만 가져옴
    logits = logits.astype(np.float32)  # float64에서 float32로 변환 (exp 오버플로우 방지)

    # 1. Temperature 적용
    if temperature == 0:  # 그리디 디코딩
        return np.argmax(logits).item()

    # logits에 마스킹 (pad_token, eos_token)
    # 이미 모델 내부에서 처리되거나, 로짓에서 직접 마스킹할 필요 없을 수 있지만, 안전을 위해.
    # Llama 3는 EOS 토큰으로 종료되므로, EOS를 마스킹하지 않습니다.
    # tokenizer.pad_token_id는 Llama 3에서 보통 eos_token_id와 동일합니다.
    if tokenizer.pad_token_id is not None:
        logits[tokenizer.pad_token_id] = -float('inf')

    # 소프트맥스 적용하여 확률 분포 얻기
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))

    # 2. Top-p (nucleus) 샘플링
    sorted_probs = np.sort(probs)[::-1]
    sorted_indices = np.argsort(probs)[::-1]

    cumulative_probs = np.cumsum(sorted_probs)
    # 누적 확률이 top_p를 초과하는 첫 번째 인덱스 찾기
    mask = cumulative_probs > top_p
    if np.any(mask):
        last_idx = np.where(mask)[0][0]
        sorted_probs = sorted_probs[:last_idx + 1]
        sorted_indices = sorted_indices[:last_idx + 1]

    # 확률 정규화
    sorted_probs = sorted_probs / np.sum(sorted_probs)

    # 샘플링
    next_token = np.random.choice(sorted_indices, p=sorted_probs)
    return next_token.item()


# --- 메인 추론 루프 ---
def run_inference(user_prompt):
    messages = [{"role": "user", "content": user_prompt}]

    # 1. 초기 프롬프트 토큰화 (Llama 3 챗 템플릿 적용)
    # Llama 3는 챗 템플릿을 적용한 후 add_generation_prompt=True로 하면 <|start_header_id|>assistant<|end_header_id|>\n 이 붙습니다.
    # 이 부분은 모델의 응답 시작을 알리는 역할을 합니다.
    formatted_prompt = apply_llama_chat_template(messages, add_generation_prompt=True)

    input_ids = tokenizer(formatted_prompt, return_tensors="np").input_ids.astype(np.int64)
    attention_mask = np.ones_like(input_ids).astype(np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)

    current_length = input_ids.shape[1]  # 현재까지의 전체 시퀀스 길이
    generated_tokens = input_ids[0].tolist()

    # 2. 초기 KV 캐시 (모두 0으로 초기화된 텐서)
    # float16으로 초기화해야 모델의 dtype과 맞습니다.
    past_key_values = [
        np.zeros((DUMMY_BATCH_SIZE, DUMMY_NUM_HEADS, 0, DUMMY_HEAD_DIM), dtype=np.float16)
        for _ in range(DUMMY_NUM_LAYERS * 2)  # Key, Value 쌍이므로 2배
    ]

    print("\n--- 모델 응답 ---")
    start_time = time.time()

    for _ in range(MAX_NEW_TOKENS):
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        # 이전 스텝의 present_kv를 current_past_kv로 전달
        for i, name in enumerate(past_kv_names):
            onnx_inputs[name] = past_key_values[i]

        # ONNX 추론 실행
        outputs = session.run(output_names, onnx_inputs)
        output_dict = {name: out for name, out in zip(output_names, outputs)}

        logits = output_dict["logits"]
        present_kv = [output_dict[name] for name in present_kv_names]

        # 다음 토큰 샘플링
        next_token_id = sample_next_token(logits, TEMPERATURE, TOP_P)

        # EOS 토큰이면 생성 중지
        if next_token_id == tokenizer.eos_token_id:
            break

        # 생성된 토큰 추가
        generated_tokens.append(next_token_id)

        # 새로운 입력 준비 (다음 스텝은 단일 토큰만 입력)
        input_ids = np.array([[next_token_id]], dtype=np.int64)

        # 어텐션 마스크 업데이트 (누적 길이)
        current_length += 1
        attention_mask = np.ones((1, current_length), dtype=np.int64)

        # 포지션 ID 업데이트 (현재 토큰의 위치)
        position_ids = np.array([[current_length - 1]], dtype=np.int64)

        # past_key_values 업데이트
        past_key_values = present_kv

        # 부분 디코딩 및 출력
        # print(tokenizer.decode(generated_tokens[len(formatted_prompt_tokens):], skip_special_tokens=True), end='', flush=True)
        # ^ 위 방식은 토큰 하나씩 출력하는데, 한글의 경우 글자가 깨질 수 있으니 모아서 출력하는 것이 좋습니다.

    end_time = time.time()

    # 전체 응답 디코딩 (프롬프트 제외)
    # Llama 3 템플릿이 붙기 전의 원래 프롬프트 길이를 알아야 한다.
    # 간단하게 전체 생성된 토큰에서 Llama 3 시스템/유저 프롬프트 및 어시스턴트 프롬프트 부분을 제거
    # 여기서는 Llama 3의 시스템/유저/어시스턴트 템플릿 토큰을 고려해야 하므로
    # `tokenizer.apply_chat_template`의 `tokenize=True`로 프롬프트 토큰 길이를 먼저 얻는 것이 좋다.

    # 간단한 디코딩: 전체 생성된 토큰에서 원래 프롬프트에 해당하는 부분을 제외하고 디코딩
    # 여기서는 생성된 전체 토큰에서 첫 번째 프롬프트 토큰들을 제외하고 디코딩
    # Llama 3 템플릿 (`<|start_header_id|>user<|end_header_id|>\n\n<user_message><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`) 때문에
    # 순수 응답만 분리하기가 까다로울 수 있다.
    # 여기서는 간단하게 전체 디코딩 후 마지막 토큰만 출력하는 것으로 대체

    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)  # skip_special_tokens=False로 템플릿 확인

    # Llama 3의 `apply_chat_template`에 따라 잘린 부분을 찾아 응답만 추출하는 로직이 필요할 수 있다
    # 예: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" 이후의 텍스트
    assistant_prefix = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False,
                                                     add_generation_prompt=True)

    # 응답 시작 부분 찾기
    response_start_index = full_response.find(assistant_prefix)
    if response_start_index != -1:
        # 접두사 이후의 내용만 추출
        pure_response = full_response[response_start_index + len(assistant_prefix):]
        # <|eot_id|> 토큰이나 추가적인 Llama 템플릿 토큰이 있다면 제거
        pure_response = pure_response.replace(tokenizer.eos_token, "").strip()
    else:
        pure_response = full_response.strip()  # 찾지 못하면 전체 응답 사용

    print(pure_response)
    print(f"\n[총 생성 시간: {end_time - start_time:.2f}초]")


# --- 채팅 예시 ---
if __name__ == "__main__":
    print("AI 챗봇 시작! (종료하려면 'exit' 입력)")
    while True:
        user_input = input("\n당신: ")
        if user_input.lower() == 'exit':
            print("챗봇 종료.")
            break

        run_inference(user_input)





# $env:PYTHONPATH = (Get-Location).Path
# $env:PATH += ";C:\Program Files\NVIDIA\CUDNN\v9.0\bin\12.3"
# $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
# python model/nvidia_onnx_model_run.py

