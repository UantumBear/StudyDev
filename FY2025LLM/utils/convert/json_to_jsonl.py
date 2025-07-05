import json
import argparse
from pathlib import Path

def convert_json_to_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"변환 완료: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 .json 파일 경로")
    parser.add_argument("--output", required=True, help="출력 .jsonl 파일 경로")
    args = parser.parse_args()

    convert_json_to_jsonl(args.input, args.output)
    """ $ 명령어
    python utilities/convert/json_to_jsonl.py \
      --input data/DevBear/llama3_train.json \
      --output data/converted/DevBear/llama3_train.jsonl
    """


