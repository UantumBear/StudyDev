이 경로는 모델을 파인튜닝 / 경량화 / 추론 등을 수행할 경로입니다.

기능을 utils 에서 관리하지만 모델의 경우 model 경로에서 관리합니다.

models: 원본 모델 경로
model : 최종 모델 경로

###### model/ 하위 경로 파일 설명
| 경로                    | 설명                                            |
|-----------------------|-----------------------------------------------|
| fintuning.py          | llama Base 모델을 파인튜닝하는 소스 (LoRA어댑터 튜닝)         |
| fintuned_model_run.py | llama Base 모델을 LoRA 파인튜닝한 최종 모델을 실행하는 소스      |
| llama3_hf_download.py | llama3 계열 모델을 Huggingface 에서 로컬로 다운로드 하는 소스   | 
| llama_chat_model.py   | llama3-instruction 계열의 최종 모델을 사용하기 위한 소스 (추론) |
| run.py                | 각종 테스트 실행 소스                                  |
