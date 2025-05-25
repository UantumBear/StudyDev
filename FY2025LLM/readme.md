
## Study

#### Settings 
C:\Users\litl\AppData\Local\Programs\Python\Python313\python.exe -m venv venv
python version: 3.13
pip freeze > FY2025LLM/requirements.txt


#### 0. 모델 다운로드를 위한 llama 페이지에서의 url 발급
https://www.llama.com/llama-downloads/ 
 
Requested models:
- Llama 3.3: 70B  
- Llama 3.2: 1B & 3B  
- Llama 3.2: 11B & 90B  
- Llama 3.1: 8B & 405B  

---  

#### 1. Llama CLI 설치하기
```shell
pip install llama-stack
```
```shell
# pip install llama-stack # 이전 버전이 있었을 경우 업그레이드
```

#### 2. 내 nvidia gpu 확인하기
```shell
nvidia-smi
```
결과
```shell
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   44C    P3             12W /   40W |       0MiB /   6141MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

#### 3. 모델 리스트 확인하기 
```shell
llama download --model llama3-2-3b
```
결과
```shell
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Model Descriptor(ID)                    ┃ Hugging Face Repo                                   ┃ Context Length ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Llama3.1-8B                             │ meta-llama/Llama-3.1-8B                             │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-70B                            │ meta-llama/Llama-3.1-70B                            │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-405B:bf16-mp8                  │ meta-llama/Llama-3.1-405B                           │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-405B                           │ meta-llama/Llama-3.1-405B-FP8                       │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-405B:bf16-mp16                 │ meta-llama/Llama-3.1-405B                           │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-8B-Instruct                    │ meta-llama/Llama-3.1-8B-Instruct                    │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-70B-Instruct                   │ meta-llama/Llama-3.1-70B-Instruct                   │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-405B-Instruct:bf16-mp8         │ meta-llama/Llama-3.1-405B-Instruct                  │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-405B-Instruct                  │ meta-llama/Llama-3.1-405B-Instruct-FP8              │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.1-405B-Instruct:bf16-mp16        │ meta-llama/Llama-3.1-405B-Instruct                  │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-1B                             │ meta-llama/Llama-3.2-1B                             │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-3B                             │ meta-llama/Llama-3.2-3B                             │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-11B-Vision                     │ meta-llama/Llama-3.2-11B-Vision                     │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-90B-Vision                     │ meta-llama/Llama-3.2-90B-Vision                     │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-1B-Instruct                    │ meta-llama/Llama-3.2-1B-Instruct                    │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-3B-Instruct                    │ meta-llama/Llama-3.2-3B-Instruct                    │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-1B-Instruct:int4-qlora-eo8     │ meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8     │ 8K             │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-1B-Instruct:int4-spinquant-eo8 │ meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8 │ 8K             │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-3B-Instruct:int4-qlora-eo8     │ meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8     │ 8K             │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-3B-Instruct:int4-spinquant-eo8 │ meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8 │ 8K             │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-11B-Vision-Instruct            │ meta-llama/Llama-3.2-11B-Vision-Instruct            │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.2-90B-Vision-Instruct            │ meta-llama/Llama-3.2-90B-Vision-Instruct            │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama3.3-70B-Instruct                   │ meta-llama/Llama-3.3-70B-Instruct                   │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-4-Scout-17B-16E                   │ meta-llama/Llama-4-Scout-17B-16E                    │ 256K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-4-Maverick-17B-128E               │ meta-llama/Llama-4-Maverick-17B-128E                │ 256K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-4-Scout-17B-16E-Instruct          │ meta-llama/Llama-4-Scout-17B-16E-Instruct           │ 10240K         │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-4-Maverick-17B-128E-Instruct      │ meta-llama/Llama-4-Maverick-17B-128E-Instruct       │ 1024K          │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-4-Maverick-17B-128E-Instruct:fp8  │ meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8   │ 1024K          │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-4-12B                       │ meta-llama/Llama-Guard-4-12B                        │ 8K             │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-3-11B-Vision                │ meta-llama/Llama-Guard-3-11B-Vision                 │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-3-1B:int4                   │ meta-llama/Llama-Guard-3-1B-INT4                    │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-3-1B                        │ meta-llama/Llama-Guard-3-1B                         │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-3-8B                        │ meta-llama/Llama-Guard-3-8B                         │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-3-8B:int8                   │ meta-llama/Llama-Guard-3-8B-INT8                    │ 128K           │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────┼────────────────┤
│ Llama-Guard-2-8B                        │ meta-llama/Llama-Guard-2-8B                         │ 4K             │
└─────────────────────────────────────────┴─────────────────────────────────────────────────────┴────────────────┘

```


```shell
llama model list
llama model list --show-all
```

#### 4. 모델 다운로드 하기
```shell
llama model download --source meta --model-id  MODEL_ID
llama model download --source meta --model-id Llama3.2-3B
```
위 명령어를 입력하면 다운로드를 위한 custom URL 을 입력해달라는 창이 뜬다.  
0번 에서 정보를 입력하고 임시발급된 url 을 입력한다.
```shell
# 결과
Downloading checklist.chk       ... 156 bytes
Downloading tokenizer.model     ... 2.2 MB
Downloading params.jsom         ... 220 bytes
Downloading consolidated.00.pth ... 6.4GB
Successfully downloaded model to C:\Users\litl\.llama\checkpoints\Llama3.2-3B
View MD5 checksum files at: C:\Users\litl\.llama\checkpoints\Llama3.2-3B\checklist.chk
[Optionally] To run MD5 checksums, use the following command: llama model verify-download --model-id Llama3.2-3B
```

#### 5. 모델이 다운로드 되었는지 확인하기.
```text
C:\Users\litl\.llama\checkpoints\Llama3.2-3B
checklist.chk
consolidated.00.pth
params.json
tokenizer.model
```

#### 6. 라마 모델을 허깅페이스 구조로 변환하기. (공식 convert 소스 이용)
```shell
(venv) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev>
  
python FY2025LLM/utils/huggingface/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py `
  --input_dir "C:/Users/litl/.llama/checkpoints/Llama3.2-3B" `
  --model_size 3B `
  --llama_version 3.2 `
  --output_dir "FY2025LLM/models/llama3.2-3B-hf" `
  --safe_serialization
```
※ 참고로, nvidia 온라인 교육에서 받았던 convert_llama_weights_to_hf.py 를 실행하자, 아래와 같은 에러 문구(invalid)를 받았다.
```shell
usage: convert_llama_weights_to_hf.py [-h] [--input_dir INPUT_DIR] [--model_size {7B,7Bf,13B,13Bf,30B,34B,65B,70B,70Bf,tokenizer_only}] [--output_dir OUTPUT_DIR]
                                      [--safe_serialization SAFE_SERIALIZATION]
convert_llama_weights_to_hf.py: error: argument --model_size: invalid choice: '3B' (choose from 7B, 7Bf, 13B, 13Bf, 30B, 34B, 65B, 70B, 70Bf, tokenizer_only)
```

실습에 사용했던 모델과 소스는 이전 버전이라 그렇다. llama 공식 페이지에서 최신 모델을 받았음으로  
그에 맞는 최신 convert 소스를 받아야 한다.   '
링크: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py  

---
1일차, 모델을 돌려보며,
- LLaMA 3.2 모델은 대화형 prompt 포맷이 필요하다. 
- Instruct 버전이 아닌 **Base 모델(Llama3.2-3B)**은 instruction tuning이 되어 있지 않기 때문.
- huggingface 에서 한국어 챗봇을 위한 데이터 셋을 다운받아 파인튜닝을 해보자. 
- Top-k / Top-p / Temperature 설정 
- EOS 토큰 설정


#### 7. 챗봇형 한국어 데이터셋 선택하기.
일단 허깅페이스에서 koInstruction 데이터 를 다운 받아본다. (CarrotAI/ko-instruction-dataset)

※  "messages" 필드는 반드시 system → user → assistant 순으로 포함되어야 하며,
LLaMA 3 모델은 이러한 multi-turn chat 형식의 데이터를 학습하는 데 최적화되어 있다고 한다.