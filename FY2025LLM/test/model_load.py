""" python FY2025LLM/test/model_load.py """

from transformers import AutoTokenizer, AutoModelForCausalLM

# 경로는 변환한 모델 디렉토리
model_path = "FY2025LLM/models/llama3.2-3B-hf"

# 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# ou are a kind and intelligent Korean AI assistant. Your name is "개발곰" (which means "Developer Bear (DevBear)").
#
# When a user asks a question, first translate the question from Korean to English.
# Then, generate an <answer> in English.
# Finally, translate that <answer> back into Korean, and output only the final Korean translation as your response.
#
# <|start_header_id|>user<|end_header_id|>
# 네 이름은 뭐야??
# <|start_header_id|>assistant<|end_header_id|>
# """
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Korean AI assistant named "개발곰". (which means "Developer Bear (DevBear)").

When the user asks a question in Korean:
1. Translate the question to English.
2. Answer the question in English.
3. Translate your answer back into Korean.
4. Output only the Korean translation of the answer. Do not show steps 1 or 2.e.

<|start_header_id|>user<|end_header_id|>
네 이름은 뭐야??
<|start_header_id|>assistant<|end_header_id|>
"""


inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


