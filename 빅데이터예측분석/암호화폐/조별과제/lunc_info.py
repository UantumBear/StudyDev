import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm

# Step 한글 폰트 경로 지정
font_path = 'fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path, size=8) # 폰트 프로퍼티 설정
# matplotlib의 rcParams 설정을 통해 전역적으로 한글 폰트 적용
plt.rcParams['font.family'] = font_prop.get_name()


# Step 데이터 불러오기 (원본 데이터)
file_path = 'data/output/02_cluster_0_for_train.xlsx'
df = pd.read_excel(file_path)

# 날짜 열 정렬 및 저장
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')  # 날짜 순 정렬
dates = df['Date']  # 날짜 저장

# 타겟 컬럼 설정
target_column = "LUNC-USD_Close"

# 원본 데이터 분포 확인
df[target_column].hist(bins=50)
plt.title("LUNC Close 데이터 분포")
plt.savefig('data/model/02_LUNC원본데이터분포.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 로그 변환 수행
df[target_column + "_log"] = np.log1p(df[target_column])
print(f"로그 변환된 값 (상위 10개):\n{df[target_column + '_log'].head(10)}")

# Step MinMax 정규화 수행
scaler = MinMaxScaler(feature_range=(0, 1))
df[target_column + "_scaled"] = scaler.fit_transform(df[[target_column + "_log"]])
print(f"정규화된 값 (상위 10개):\n{df[target_column + '_scaled'].head(10)}")

# Step 역정규화 + 역로그 변환 함수
def inverse_transform(scaled_values, scaler_target, log_transformed=True):
    # 정규화 역변환
    inverse_values = scaler_target.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
    # 로그 변환 복구
    if log_transformed:
        inverse_values = np.expm1(inverse_values)
    return inverse_values

# 역정규화 + 역로그 변환 테스트
df[target_column + "_restored"] = inverse_transform(
    df[target_column + "_scaled"].values, scaler, log_transformed=True
)

# 원본 값과 복구된 값 비교
print(f"원본 값 (상위 10개):\n{df[target_column].head(10)}")
print(f"복구된 값 (상위 10개):\n{df[target_column + '_restored'].head(10)}")

# 복구된 값과 원본 값 비교 시각화
plt.figure(figsize=(10, 5))
plt.plot(df['Date'],df[target_column], label="Original")
plt.plot(df['Date'],df[target_column + "_restored"], label="Restored", linestyle='dashed')
plt.legend()
plt.title("Original vs Restored Values")
plt.savefig('data/model/02_LUNC원본데이터그래프.png', dpi=300, bbox_inches='tight')
# plt.show()

# 원본과 복구된 값의 차이 확인
df["Difference"] = df[target_column] - df[target_column + "_restored"]
print(f"차이 (상위 10개):\n{df['Difference'].head(10)}")
print(f"차이의 절대평균: {df['Difference'].abs().mean()}")
