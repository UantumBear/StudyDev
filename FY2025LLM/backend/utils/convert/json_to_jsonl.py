import json
import argparse
from pathlib import Path

def convert_json_to_jsonl(input_path, output_path, system_path=None):
    # system 메시지를 미리 불러오기
    system_message = None
    if system_path:
        with open(system_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            system_message = {"role": "system", "content": content}

    # 원본 JSON 읽기
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # .jsonl 변환 + system message 삽입
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in data:
            messages = sample.get("messages", [])
            if system_message:
                messages = [system_message] + messages
                sample["messages"] = messages
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"변환 완료: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 .json 파일 경로")
    parser.add_argument("--output", required=True, help="출력 .jsonl 파일 경로")
    parser.add_argument("--system", required=False, help="system prompt 텍스트 파일 경로")
    args = parser.parse_args()

    convert_json_to_jsonl(args.input, args.output, args.system)
    """ 
    [PowerShell] 명령어
    python utils/convert/json_to_jsonl.py `
      --input data/DevBear/llama3_train.json `
      --output data/converted/DevBear/llama3_train.jsonl `
      --system data/DevBear/system_prompt.txt
    """


