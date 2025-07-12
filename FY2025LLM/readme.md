## FY2025LLM

2025년 llm을 공부하는 경로
```bash
echo "# Generated on $(Get-Date -Format 'yyyy-MM-dd')" > requirements.txt; pip freeze >> requirements_loc_windows_venv312.txt
```

!! Instruction 모델로 드디어 개발곰의 이름을 받았다!
![data/DevBear/png/llama3-instruction-devbear0001.png](data/DevBear/png/llama3-instruction-devbear0001.png)




## Study

#### Settings 
C:\Users\litl\AppData\Local\Programs\Python\Python312\python.exe -m venv venv312`
python version: 3.12 
현재 3.13 은 CUDA 지원이 안되므로 3.12로 가상환경을 생성한다. (25.05.25)
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
llama model download --source meta --model-id Llama3.2-1B
```
위 명령어를 입력하면 다운로드를 위한 custom URL 을 입력해달라는 창이 뜬다.  
0번 에서 정보를 입력하고 임시발급된 url 을 입력한다.
3B가 CUDA Memory Out 이 나서 1B로 재 다운로드 받았다.
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

2일차, 파인튜닝을 돌려보며,
- VRAM 6GB 로는 3B는 턱없이 부족하다. (float 16, max_length, batch 조절)
- 1B 는 float16에서 파인튜닝이 돌아간다.
- loss:nan 발생으로 dataset 과 padding 부를 보고 있다.  

3일차, 파인튜닝과 추론을 돌려보며,  
기본 Llama Base Model 은 사전학습된 모델로, 챗봇 역할을 수행하려면  
instruction following 능력 OR 대화형 응답 능력을 갖추도록 파인튜닝 하여야 한다. 
기본 torch 로 직접 train 을 하자 계속해서 Loss: nan 이 발생하여 SFTTrainer() 로 튜닝을 해보았다.  
학습이 실행 자체는 되나, tokenizer 와 model 의 size 가 맞지 않았다.  
coverted 경로에 tokenizer.model 이 없어서, llama->hf 변환 후 직접 복붙해주었었는데,  
튜닝 후에도 tokenizer.model 이 저장 되지 않았다.  
tokenizer.json 을 보니 PretrainedTokenizer 로 설정되어있는데, 해당 변환 파일도 조금 수정 후 다시 클래스 이름을 맞춰서  
진행해볼까 한다.  


파인튜닝 기법은 크게 아래와 같은 것들이 있다.  
##### SFT (Supervised Fine-Tuning)
- 역할 지시 + 질문 + 답변 패턴을 학습
- [{"role": ..., "content": ...}, ...] 구조의 chat 형식 데이터
- SFTTrainer, HuggingFace Trainer  
- 챗봇의 성격을 형성하는 핵심 단계
- "system/user/assistant" 역할로 구성된 대화 데이터를 넣고, assistant의 답변을 학습한다.

##### PEFT (LoRA, QLoRA, LoHa, LOLA 등)
- 파인 튜닝 시, 일부 파라미터만 업데이트 하는 기법

##### RLHF (Reinforcement Learning with Human Feedback)
- 모델의 응답을 더 잘 정제하기 위한, 보상 모델 학습
- trl, accelerate, reward_model, PPOTrainer

#### 7. 챗봇형 한국어 데이터셋 선택하기.
일단 허깅페이스에서 koInstruction 데이터 를 다운 받아본다. (CarrotAI/ko-instruction-dataset)

※  "messages" 필드는 반드시 system → user → assistant 순으로 포함되어야 하며,
LLaMA 3 모델은 이러한 multi-turn chat 형식의 데이터를 학습하는 데 최적화되어 있다고 한다.

그리고 해당 데이터를 llama3 구조에 맞게 변환시켜 준다.


#### 8. 파인튜닝 하기.
쿠다 사용 가능 여부 확인
```shell
nvidia-smi
# CUDA Version: 12.6 
python -c "import torch; print(torch.cuda.get_device_name(0))"
# False
```
라이브러리 설치
```shell
pip uninstall torch -y
(venv) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev> 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 결과
Looking in indexes: https://download.pytorch.org/whl/cu121, https://pypi.ngc.nvidia.com
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
ERROR: No matching distribution found for torch
```
3.13 CUDA 지원이 안되는 듯 하다. 3.12로 가상환경을 만들자.

라이브러리 설치 후 인덱싱 중 wsl 을 사용하기 위한 구성 요소를 확인해달라는 창이 떴다.
전에 파인튜닝했을때는 이런 것이 뜨지않았는데!?

일단 실행시켜주었다. 아 가상환경잡을때 뜬것같은데 TODO 나중에확인하기.
```shell
# C:\Program Files\WSL\wsl.exe
Windows 선택적 구성 요소 VirtualMachinePlatform 설치  
배포 이미지 서비스 및 관리 도구  
버전: 10.0.26100.1150  
이미지 버전: 10.0.26100.4061
```

```shell
(venv312) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev>
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 결과
NVIDIA GeForce RTX 4050 Laptop GPU

```

TODO 별도의 finetuned 경로를 만들어서
파인튜닝한 모델 + 변경한 토크나이저 + 체크포인트 저장하기

TODO
Error 
```shell
(1)
RuntimeError: 
CUDA error: 
CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx( 
handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16F, lda, b, CUDA_R_16F, ldb, &fbeta, c, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
(2) 메모리 부족
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```


###### 25.07.12 완성된 onnx 모델을 다운받아 사용해보기.

##### Nvidia CLI 다운받기
https://org.ngc.nvidia.com/setup/installers/cli
C:\Users\litl\설치파일\file\setup.exe 실행
C:\Users\litl\설치파일\file\ngccli\x86\ngc.exe registry model download-version nvidia/meta-llama-3.2-3b-onnx-int4-rtx:1.0
-> 다운로드
models/nvidia/meta-llama-3.2-3b-onnx-int4-rtx_v1.0/llama32_onnx_int4_genai_dml

```bash
[ERROR] cudnn 을 필요로 함
상세:
2025-07-12 21:40:37.7449497 [E:onnxruntime:Default, provider_bridge_ort.cc:2195 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1778 onnxru
ntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : Error loading "C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM\venv312\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll" which depends on "cudnn64_9.dll" which is missing. (Error 126: "읠?

[해결]
환경 변수에 cudnn 경로를 등록
```

##### TensorRT SDK 다운로드