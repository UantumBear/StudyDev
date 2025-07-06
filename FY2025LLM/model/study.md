### Epoch 란?

전체 학습 데이터를 한 번 모두 사용해서 학습한 것.  
예를 들어, dataset 에 샘플이 1000개 있다면, 1 epoch == 1000개를 모두 모델에 한번 보여준 것을 말한다.  

### Step 이란?

한 batch 를 학습하는 것을 1 step 이라고 한다.  
예를 들어, 총 1000개를 학습하려면, batch_size=100 으로 설정할 때, 총 10step 이 필요한 것이다.  

### gradient_accumulation_steps 란?
일반적으로 모델은 N개의 데이터를 한 번에 넣고,  loss 계산 + 역전파를 수행한다.   
그런데 VRAM이 부족하면, 한 번에 큰 배치(batch) 를 넣을 수 없다.  
그 때, Gradient Accumulation 를 사용한다.  

예를 들어, 한번에 batch_size=8 로 학습시키고 싶지만, VRAM 이 부족하다면,
per_device_train_batch_size = 2 (한 번에 GPU에 올리는 샘플 수)  
gradient_accumulation_steps = 4 (몇 step 동안 gradient를 누적할지)  
-> 2개 씩, 4번 이렇게 하는 것이다.  

CUDA OOM (out-of-memory) 에러가 난다면,  
모델이 처리하는 실제 배치 크기는 유지하면서, (8)
한 번에 GPU에 올리는 데이터 수 (2) 를 줄였기 때문에,  VRAM 사용량은 감소하게 된다. 

--- 

### TrainingArguments 속성

#### bf16
``` bf16=is_bf16_supported ```  
AMP(자동 혼합 정밀도)에서 bfloat16 사용 여부를 말한다.
Ampere(RTX 30xx, A100 등) 이상의 GPU에서 지원된다고 한다. 
FP16보다 안정적이라고 한다.  

#### save_total_limit
``` save_total_limit=1 ```
저장하는 체크포인트 개수. 1라는 것은, 최신 1개만 저장한다는 것을 말한다.

#### load_best_model_at_end
``` load_best_model_at_end=True ```
학습 종료 후 평가 지표가 가장 좋았던 모델로 자동으로 불러오는 옵션

---

### LoraConfig

#### r 
```Low-Rank Adaption에서의 랭크(rank)```   
LoRA는 큰 weight matrix를 작은 두 개의 저차 행렬로 분해해 학습한다.  
r이 클수록 학습 파라미터 수는 증가하지만 표현력은 좋아진다.  
보통 4~128 사이의 값을 사용한다고 한다.  

#### lora_alpha
```scaling factor (확대 계수)```
학습되는 LoRA 레이어의 출력을 얼마나 크게 반영할지를 결정한다. 

#### task_type
task_type이란 현재 모델이 수행할 task의 종류를 말한다.
```task_type="CAUSAL_LM"```
CAUSAL_LM : GPT 계열 모델처럼 한 방향(좌->우) 으로 텍스트를 생성하는 언어 모델  
SEQ_CLS   : 시퀀스 분류 (예: 감정 분류)  
TOKEN_CLS : 토큰 분류 (예: 개체명 인식)  

--- 

### chat_templates 속성
#### add_generation_prompt
```add_generation_prompt=True``` 속성을 쓰면,   
SFTTrainer는 assistant의 실제 답변 부분이 아닌, 생성 프롬프트(assistant 헤더 토큰) 이후의 내용을 생성하도록 학습하게 된다.  
이는 모델이 실제로 "제 이름은 개발곰입니다."와 같은 완전한 답변을 생성하는 패턴을 학습하는 것을 방해할 수 있다.  

### .
#### bitsandbytes 라이브러리
윈도우 환경에서는 bitsandbytes를 사용한 4비트 양자화가 어렵다.
bf16/fp16을 사용한다.

### SFTTrainer
#### formatting_func
데이터 포매팅 함수를 동적으로 적용





