import os
from hwp5 import hwp5proc
print(dir(hwp5proc))
print("=================================================================================================")
"""
위 hwp5 라이브러리는 pip install pyhwp 로부터 온다.
print(dir(hwp5proc)) 를 통해, 실제 라이브러리 내의 함수들을 살펴보자.

pip install pyhwp 로 받았던 라이브러리 print 결과:
['ArgumentParser', 'InvalidHwp5FileError', 'PY3', 'ParseError', '_', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', 'absolu
te_import', 'copyright', 'disclosure', 'gettext', 'init_logger', 'license', 'locale_dir', 'logger', 'logging', 'main', 'main_argparser', 'os', 'print_function', 'program', 'sys', 't', 'unicode_literals', 'version']

이 pyhwp 라이브러리는 CLI 전용 도구라고 한다. 
즉 python 에서 실행하고 싶다면, CLI 명령어를 수행하도록 소스코드를 작성해야 한다.
 
"""

################################# CLI 명령어를 사용한 pyhwp 를 통한 문서 변환 #################################
import subprocess
# # 실행 파일 경로
current_file_path = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(current_file_path)
print(CURRENT_DIR)
# 파일 경로
HWP_FILE_NAME = "아래아21_테스트01.hwp"
HWP_FILE_PATH = os.path.join(CURRENT_DIR, "data", HWP_FILE_NAME)
print(f"HWP_FILE_PATH: {HWP_FILE_PATH}")
def extract_text_via_hwp5txt(hwp_path: str) -> str:
    try:
        result = subprocess.run(
            ['hwp5txt', hwp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("[ERROR] 명령어 실행 실패:", e.stderr)
        return ""
    except FileNotFoundError:
        print("[ERROR] 'hwp5txt' 명령어를 찾을 수 없습니다. PATH 설정 또는 설치 확인 필요.")
        return ""

if __name__ == "__main__":

    text = extract_text_via_hwp5txt(HWP_FILE_PATH)
    if text:
        print("=== 추출된 텍스트 ===")
        print(text)

        # 텍스트 파일로 저장
        output_path = HWP_FILE_PATH.replace('.hwp', '.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[INFO] 저장 완료: {output_path}")

"""
실행 결과:
[ERROR] 명령어 실행 실패: Not an OLE2 Compound Binary File.

원인:
.hwp 파일이 HWP 5.0 (OLE2 기반 포맷)이 아니기 때문에 발생한다.
 내가 이 hwp를 만들 떄 docx -> hwp 변환에서, 구버전 hwp(==아래아 한글 2.0) 를 선택했는데, 그래서 그런 듯 하다.
 
 pyhwp 또는 hwp5txt CLI 도구는 HWP 5.0 이상의 문서 포맷만 지원한다고 한다. 일단 hwp 파일 처리를 해야 하므로,
 구버전이 아닌 최근 버전 (아래아 한글 2.1/2.5) 한글 파일로 테스트를 해보자.
 구버전은 일단 예외 처리만 해두도록 하자.
 
"""

""" 일단 파일의 정확한 형식을 알아보자. 
HWP 포맷에는 시그니처 라는 것이 있다고 한다. 
포맷	             /   Signature (Hex)	     / 의미
HWP 5.0 (OLE2)	 / d0 cf 11 e0 a1 b1 1a e1	 / Microsoft Compound File (OLE2)
HWP 3.x 이하	     / 48 57 50 20 (HWP )	     / ASCII 기반
HWPX	         / ZIP 구조 (50 4B 03 04)	 / zip 기반 OOXML 유사 구조

"""
import olefile

def get_file_signature(path: str, num_bytes=8):
    with open(path, 'rb') as f:
        sig = f.read(num_bytes)
        return sig.hex(' ').upper()

def detect_hwp_format(path: str):
    signature = get_file_signature(path)

    if signature.startswith("D0 CF 11 E0 A1 B1 1A E1"):
        print("[INFO] OLE2 기반 HWP 5.0 이상 포맷입니다.")
    elif signature.startswith("48 57 50 20"):
        print("[INFO] HWP 3.x 이하 (구버전 포맷)입니다.")
    elif signature.startswith("50 4B 03 04"):
        print("[INFO] HWPX 포맷 (ZIP 기반)입니다.")
    else:
        print("[WARN] 알 수 없는 HWP 포맷입니다.")
    print(f"[DEBUG] 파일 시그니처: {signature}")

if __name__ == "__main__":
    detect_hwp_format(HWP_FILE_PATH)
# Pycharm Terminal (Windows): python utilities/elastic/test_hwp.py