"""
$ python FY2025LLM/test/dataset_download.py

"""

from datasets import load_dataset
import json
import os

dataset = load_dataset("CarrotAI/ko-instruction-dataset", split="train")

save_path = "FY2025LLM/data/CarrotAI"
os.makedirs(save_path, exist_ok=True)

with open(f"{save_path}/ko_instruction.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print("✅ CarrotAI 데이터셋 저장 완료")

