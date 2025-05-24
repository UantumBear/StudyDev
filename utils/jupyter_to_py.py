""" 주피터 노트북의 소스코드 부를 py 파일로 추출하는 유틸 """

import os
import nbformat
from conf import ROOT_DIR

# 입력 파일 경로
input_path = rf"{ROOT_DIR}/경로/파일명.ipynb"
# 출력 파일 경로
output_path = rf"{ROOT_DIR}/경로/파일명.py"
# 출력 폴더 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Jupyter Notebook 파일 읽기
with open(input_path, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# 코드 셀만 추출
code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
code_only = "\n\n".join(code_cells)

# .py 파일로 저장
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(code_only)

print(f"코드가 {output_path} 에 저장되었습니다.")


