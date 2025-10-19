import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

"""  이것은 단순히 전체 정규화된 데이터를 가지고 표준화 후 PCA 하는 과정 """

# 데이터 로드
data_path = 'data/rows_1825_with_normalize.xlsx'  # 실제 데이터 경로로 변경
data = pd.read_excel(data_path)  # pd.read_excel을 사용하여 Excel 파일 로드
# 각 컬럼의 데이터 타입 확인
print(data.dtypes)

# 숫자형이 아닌 데이터 타입이 있는 컬럼 확인 및 제거
non_numeric_columns = data.select_dtypes(exclude=['number']).columns
if not non_numeric_columns.empty:
    print("\nNon-numeric columns will be removed:", non_numeric_columns)

# 'Date' 컬럼 제거
if 'Date' in data.columns:
    data = data.drop('Date', axis=1)  # 'Data' 컬럼을 데이터프레임에서 제거

# 데이터 표준화
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)


# PCA 모델 생성 및 학습 (예: 5차원으로 축소)
pca = PCA(n_components=5)
data_pca = pca.fit_transform(data_standardized)

# PCA 로딩 추출
loadings = pca.components_
loadings_df = pd.DataFrame(
    loadings.T, columns=[f'PC{i + 1}' for i in range(5)], index=data.columns
)
# 로딩 출력 및 파일로 저장
print(loadings_df)
loadings_df.to_excel('data/pca_data.xlsx', sheet_name='Loadings')
# 각 주성분별로 가장 큰 로딩 값을 가진 코인 찾기
representative_coins = {}
for pc in loadings_df.columns:
    max_loading_idx = np.argmax(np.abs(loadings_df[pc]))
    representative_coins[pc] = loadings_df.index[max_loading_idx]

# 결과 출력
print("Representative coins for each principal component:")
for pc, coin in representative_coins.items():
    print(f"{pc}: {coin}")