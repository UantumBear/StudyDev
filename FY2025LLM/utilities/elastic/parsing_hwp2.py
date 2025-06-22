import os
from hwp5.hwp5txt import extract_text

# # 실행 파일 경로
current_file_path = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(current_file_path)
print(CURRENT_DIR)
# 파일 경로
test_hwp_name = "테스트01.hwp"
test_hwp_path = f"{CURRENT_DIR}/data/{test_hwp_name}"


def extract_text_from_hwp(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        content = f.read()
    return extract_text(content)

if __name__ == "__main__":

    print(f"[INFO] 현재 스크립트 위치: {CURRENT_DIR}")
    print(f"[INFO] 대상 파일 경로: {test_hwp_path}")

    try:
        text = extract_text_from_hwp(test_hwp_path)
        print("=== 추출된 텍스트 ===")
        print(text)

        # 파일 저장
        output_path = test_hwp_path.replace(".hwp", ".txt")
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"[INFO] 텍스트가 {output_path}에 저장되었습니다.")

    except Exception as e:
        print(f"[ERROR] 텍스트 추출 중 오류 발생: {e}")


# Pycharm Terminal (Windows): python utilities/elastic/parsing_hwp2.py