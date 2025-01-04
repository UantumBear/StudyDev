import pandas as pd

# 데이터 불러오기
price_data_path = 'data/rows_1825_with_normalize.xlsx'  # 가격 데이터 파일 경로
cluster_data_path = 'data/cluster_results.csv'  # 클러스터 라벨 파일 경로

# 데이터 불러오기
cluster_data = pd.read_csv(cluster_data_path)

# 딕셔너리 생성
cluster_dict = {}
for index, row in cluster_data.iterrows():
    if row['Cluster'] not in cluster_dict:
        cluster_dict[row['Cluster']] = [row['Crypto']]
    else:
        cluster_dict[row['Cluster']].append(row['Crypto'])

# 가격 데이터 불러오기
price_data = pd.read_excel(price_data_path)

# 각 클러스터별로 데이터 추출 및 CSV 저장
# for cluster, cryptos in cluster_dict.items():
#     cluster_price_data = price_data[cryptos]  # 존재하는 코인만 포함되어야 함
#     cluster_price_data.to_csv(f'data/cluster_group_{cluster}.csv', index=False)

# Excel 파일에 클러스터별 데이터 저장
output_path = 'data/clustered_data_groupby.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for cluster, cryptos in cluster_dict.items():
        # 존재하는 코인 데이터만 필터링
        cluster_price_data = price_data.loc[:, cryptos]  # 코인 이름이 열에 있어야 함
        cluster_price_data.to_excel(writer, sheet_name=f'Cluster_{int(cluster)}', index=False)

print("클러스터별 데이터가 하나의 Excel 파일에 각 시트로 저장되었습니다.")
