from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 ID
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# save_path = "models/meta-llama/Llama-3.2-1B-Instruct"
#
model_id = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
save_path = "models/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

# 1. 모델 & 토크나이저 로드 (최초 1회만 인터넷 필요)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. 로컬 디스크로 저장
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# python model/llama3_hf_download.py