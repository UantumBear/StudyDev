"""
$ python FY2025LLM/test/convert_dataset_to_llama3.py
"""


import json
import os

input_path = "FY2025LLM/data/CarrotAI/ko_instruction.jsonl"
output_path = "FY2025LLM/data/converted/CarrotAI/ko_instruction_to_llama3_format.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

system_prompt = "당신은 친절하고 똑똑한 한국어 인공지능 비서입니다. 질문에 친절하고 명확하게 응답하세요."

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        example = json.loads(line)

        user_message = example["instruction"]
        if example.get("input"):
            user_message += "\n" + example["input"]

        llama_format = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message.strip()},
                {"role": "assistant", "content": example["output"].strip()},
            ]
        }

        json.dump(llama_format, fout, ensure_ascii=False)
        fout.write("\n")

print(f"✅ 변환 완료: {output_path}")
