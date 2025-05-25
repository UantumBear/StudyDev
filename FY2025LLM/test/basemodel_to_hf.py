"""
Llama Base 공식 모델을 huggingface 구조로 변환한다.
변환에는 huggingface/tranformers 의 변환 코드를 사용한다.
"""


# python FY2025LLM/utils/huggingface/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py `
#   --input_dir "C:/Users/litl/.llama/checkpoints/Llama3.2-3B" `
#   --model_size 3B `
#   --llama_version 3.2 `
#   --output_dir "FY2025LLM/models/llama3.2-3B-hf" `
#   --safe_serialization