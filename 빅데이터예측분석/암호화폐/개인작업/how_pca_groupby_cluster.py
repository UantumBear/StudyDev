import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
""" 이것은 n_components 를 늘려가며 PCA 를 수행한 코드 """


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

        # n_components를 1부터 5까지 시도
        for n in range(1, 6):  # n_components=1~5
            print(f"\nPerforming PCA with n_components={n} for {sheet_name}")
            pca = PCA(n_components=n)
            pca.fit(standardized_data)

            # PCA 로딩값 계산
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i + 1}' for i in range(n)],
                index=data.columns
            )

            # 주성분별 대표 코인 선정
            representative_coins = {}
            for pc in loadings.columns:
                max_loading_idx = loadings[pc].abs().idxmax()
                representative_coins[pc] = max_loading_idx

            explained_variance = pca.explained_variance_ratio_.cumsum()

            # 결과 저장
            if sheet_name not in pca_results:
                pca_results[sheet_name] = []
            pca_results[sheet_name].append({
                'n_components': n,
                'representative_coins': representative_coins,
                'explained_variance': explained_variance.tolist()
            })

# 결과 출력
print("\nPCA Results:")
for sheet, results in pca_results.items():
    print(f"\n{sheet}:")
    for result in results:
        print(f"n_components={result['n_components']}:")
        print(f"  Representative Coins: {result['representative_coins']}")
        print(f"  Explained Variance: {result['explained_variance']}")
