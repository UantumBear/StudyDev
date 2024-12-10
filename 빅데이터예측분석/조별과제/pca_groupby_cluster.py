import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 로드 (Excel 파일 경로)
file_path = 'data/clustered_data_groupby.xlsx'

# 결과 저장
pca_results = {}

# Excel 파일 읽기
with pd.ExcelFile(file_path) as xls:
    sheet_names = xls.sheet_names  # 시트 이름 목록

    for sheet_name in sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")

        # 시트 데이터 불러오기
        data = pd.read_excel(xls, sheet_name=sheet_name)

        # 결측값 처리
        data = data.dropna()

        # 'Date' 컬럼 제거
        if 'Date' in data.columns:
            data = data.drop('Date', axis=1)

        # 데이터 표준화
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)

        # PCA 수행
        pca = PCA()
        pca.fit(standardized_data)

        # PCA 로딩값 계산
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(pca.n_components_)],
            index=data.columns
        )

        # 첫 번째 주성분(PC1)에서 가장 큰 절대 로딩값을 가지는 코인을 대표로 선정
        representative_coin = loadings['PC1'].abs().idxmax()
        explained_variance = pca.explained_variance_ratio_.cumsum()[0]

        # 결과 저장
        pca_results[sheet_name] = {
            'representative_coin': representative_coin,
            'explained_variance_PC1': explained_variance
        }

    # 결과 출력
    print("\nPCA Results:")
    for sheet, result in pca_results.items():
        print(
            f"{sheet}: Representative Coin = {result['representative_coin']}, Explained Variance (PC1) = {result['explained_variance_PC1']:.2%}")