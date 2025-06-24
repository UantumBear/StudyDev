##### (venv312) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev\CodingTest\A>
##### python 250624.py
##### 1.
import kagglehub
import shutil
import os

# 1. 다운로드 (캐시에 저장됨)
path = kagglehub.dataset_download("donsalehi/auto-mpg-csv")
# 2. datasets 디렉토리 만들기 (없으면)
os.makedirs("datasets", exist_ok=True)
# 3. 파일 복사 (예: auto-mpg.csv만)
src = os.path.join(path, "auto-mpg.csv")
dst = os.path.join("datasets", "auto-mpg.csv")
shutil.copyfile(src, dst)

print("Saved to:", dst)

import pandas as pd
df = pd.read_csv("datasets/auto-mpg.csv")
print(type(df))

df.to_excel("datasets/auto-mpg.xlsx", index=False)