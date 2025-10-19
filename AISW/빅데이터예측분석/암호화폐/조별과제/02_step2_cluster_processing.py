import pandas as pd

# Step 1. 클러스터링 주성분 분석 결과 데이터를 가지고 그룹화를 위한 배열 만들기
# 엑셀 파일 로드
file_path = 'data/all_cryptos_with_pc1.xlsx'  # 파일 경로를 적절히 변경하세요
pc1_df = pd.read_excel(file_path)

# 클러스터별 코인 배열 생성
cluster_0 = pc1_df.loc[pc1_df['Cluster'] == 0, 'Crypto'].dropna().tolist()
cluster_1 = pc1_df.loc[pc1_df['Cluster.1'] == 1, 'Crypto.1'].dropna().tolist()
cluster_2 = pc1_df.loc[pc1_df['Cluster.2'] == 2, 'Crypto.2'].dropna().tolist()
cluster_0.append('Date')
cluster_1.append('Date')
cluster_2.append('Date')

# 결과 출력 (옵션)
print("Cluster 0:", cluster_0)
print("Cluster 1:", cluster_1)
print("Cluster 2:", cluster_2)

# Step 2. 전체 데이터엑셀을 각 클러스터별 엑셀로 나누기
# 전체 데이터를 로드
file_path = 'data/yfinance_results_20241127_194625.xlsx'  # 전체 데이터를 가진 엑셀 파일 경로를 변경하세요
all_data_df = pd.read_excel(file_path)


# 전체 데이터에서 클러스터별로 데이터 필터링
cluster_0_data = all_data_df[all_data_df.columns.intersection(cluster_0)]
cluster_1_data = all_data_df[all_data_df.columns.intersection(cluster_1)]
cluster_2_data = all_data_df[all_data_df.columns.intersection(cluster_2)]

# 필터링된 데이터프레임을 엑셀 파일로 저장
cluster_0_data.to_excel('data/output/02_cluster_0_data.xlsx', index=False)
cluster_1_data.to_excel('data/output/02_cluster_1_data.xlsx', index=False)
cluster_2_data.to_excel('data/output/02_cluster_2_data.xlsx', index=False)

print("클러스터별 데이터 파일이 생성되었습니다!")

# Step 3. 학습을 위한 데이터 구성
# 추가 데이터 로드
additional_data_path = 'data/02_대표코인onesheets.xlsx.'
additional_data = pd.read_excel(additional_data_path)

# 3-1. `LUNC-USD`
# 추가 데이터에서 필요한 열 선택
selected_0_columns = [
    'Date',
    'LUNC-USD_Close','LUNC-USD_Open', 'LUNC-USD_Low', 'LUNC-USD_High', 'LUNC-USD_Volume',
    'KRW-Exchange-Rate_Open', 'KRW-Exchange-Rate_Close', 'KRW-Exchange-Rate_High', 'KRW-Exchange-Rate_Low'
]
selected_0_data = additional_data[selected_0_columns]
# 두 데이터 병합 (Date를 기준으로)
merged_0_data = pd.merge(cluster_0_data, selected_0_data, on='Date', how='inner')
# 결과 저장
output_0_path = 'data/output/02_cluster_0_for_train.xlsx'
merged_0_data.to_excel(output_0_path, index=False)
print(f"파일이 저장되었습니다: {output_0_path}")


# 3-2. `DASH-USD`
# 추가 데이터에서 필요한 열 선택
selected_1_columns = [
    'Date',
    'DASH-USD_Close','DASH-USD_Open', 'DASH-USD_Low', 'DASH-USD_High', 'DASH-USD_Volume',
    'KRW-Exchange-Rate_Open', 'KRW-Exchange-Rate_Close', 'KRW-Exchange-Rate_High', 'KRW-Exchange-Rate_Low'
]
selected_1_data = additional_data[selected_1_columns]
# 두 데이터 병합 (Date를 기준으로)
merged_1_data = pd.merge(cluster_1_data, selected_1_data, on='Date', how='inner')
# 결과 저장
output_1_path = 'data/output/02_cluster_1_for_train.xlsx'
merged_1_data.to_excel(output_1_path, index=False)
print(f"파일이 저장되었습니다: {output_1_path}")


# 3-3. `DASH-USD`
# 추가 데이터에서 필요한 열 선택
selected_2_columns = [
    'Date',
    'BSV-USD_Close','BSV-USD_Open', 'BSV-USD_Low', 'BSV-USD_High', 'BSV-USD_Volume',
    'KRW-Exchange-Rate_Open', 'KRW-Exchange-Rate_Close', 'KRW-Exchange-Rate_High', 'KRW-Exchange-Rate_Low'
]
selected_2_data = additional_data[selected_2_columns]
# 두 데이터 병합 (Date를 기준으로)
merged_2_data = pd.merge(cluster_2_data, selected_2_data, on='Date', how='inner')
# 결과 저장
output_2_path = 'data/output/02_cluster_2_for_train.xlsx'
merged_2_data.to_excel(output_2_path, index=False)
print(f"파일이 저장되었습니다: {output_2_path}")