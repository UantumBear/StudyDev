"""
@경로 utils/convert/jsonl_merge_jsonl.py
@목적
models/llama3.2-1B-hf/ 하위의 llama3 Base 모델을 챗봇형 데이터로 파인튜닝하기 위해,
사용하는 jsonl 데이터 들을 한 데 합치는 역할
"""

import argparse
from pathlib import Path

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in input_files:
            path = Path(file_path)
            if not path.exists():
                print(f"[FAIL] 파일이 존재하지 않습니다: {file_path}")
                continue

            with open(path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():  # 빈 줄 제거
                        outfile.write(line)
        print(f"[SUCCESS] 병합 완료: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="여러 .jsonl 파일을 하나로 병합합니다.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="입력할 .jsonl 파일들의 경로 리스트 (공백으로 구분)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="병합된 결과를 저장할 .jsonl 파일 경로"
    )
    args = parser.parse_args()

    merge_jsonl_files(args.inputs, args.output)

"""
[PowerShell] 
python utils/convert/jsonl_merge_jsonl.py `
  --inputs data/converted/DevBear/llama3_train.jsonl data/converted/CarrotAI/cleaned_llama3_dataset.jsonl `
  --output data/converted/merged/train_total.jsonl

"""