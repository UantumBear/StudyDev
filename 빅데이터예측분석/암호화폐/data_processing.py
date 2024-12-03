import pandas as pd
from sklearn.preprocessing import MinMaxScaler # 정규화를 위함
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from scipy.cluster.hierarchy import fcluster # 계층적 군집화(덴드로그램)를 위함
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm

# 한글 폰트 경로 지정
font_path = 'data/fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path, size=8) # 폰트 프로퍼티 설정
# matplotlib의 rcParams 설정을 통해 전역적으로 한글 폰트 적용
plt.rcParams['font.family'] = font_prop.get_name()


# Step 1. CSV 파일 읽기
df = pd.read_excel('data/yfinance_results_20241127_194625.xlsx')

# 데이터 확인하기
valid_counts = df.notnull().sum() # 각 열의 유효한 데이터 수 세기
sorted_counts_asc = valid_counts.sort_values() # count 값 기준으로 오름차순 정렬, Series 객체를 정렬한 복사본을 반환
sorted_counts_desc = valid_counts.sort_values(ascending=False) # count 값 기준으로 내림차순 정렬

# 데이터 확인 결과 출력
pd.set_option('display.max_rows', None) # 생략 없이 모든 행 출력 설정
print(sorted_counts_asc)

# Step 2. 데이터 선택하기
columns_rows_1825 = valid_counts[valid_counts == 1825].index # 데이터 개수가 1825개인 열만 선택
filtered_df_rows_1825= df[columns_rows_1825].copy() # 원본 데이터프레임에서 해당 열만 선택, 복사하여 반환
# 선택한 데이터 확인 결과 출력
num_columns = filtered_df_rows_1825.shape[1] # filtered_df가 가진 열 개수 확인
print(f"filtered_df에는 {num_columns}개의 열이 있습니다.")
# print(filtered_df) # 전체 데이터 출력
# Excel 파일로 저장
filtered_df_rows_1825.to_excel('data/output/rows_1825.xlsx', index=False)

# Step 3. 정규화 하기
# 정규화할 데이터프레임
target_normalize_df = filtered_df_rows_1825.copy()  # 행: 1825개, 열: 133개
# 정규화 전 결측치 재확인
missing_values_count = target_normalize_df.isnull().sum() # 각 열별 결측치 개수
print(f"missing_values_count: \n {missing_values_count}")

# date 열 분리
date_column = target_normalize_df['Date']  # date 열 따로 저장
# 나머지 숫자형 열만 선택
numeric_df = target_normalize_df.drop(columns=['Date'])  # date 열 제외
numeric_df = numeric_df.select_dtypes(include=['float64', 'int64'])  # 숫자형 열만 선택
print(f"정규화 대상 열 개수: {numeric_df.shape[1]}") # Date 열 제외한 숫자형 열의 개수 확인
# MinMaxScaler로 정규화
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_df)
# 정규화된 데이터프레임 생성
normalized_df_rows_1825 = pd.DataFrame(normalized_data, columns=numeric_df.columns)
normalized_df_rows_1825['Date'] = date_column.values # date 열 다시 합치기
print(normalized_df_rows_1825.head()) # 결과 확인


# Date 열을 맨 앞으로 이동
normalized_df_rows_1825 = normalized_df_rows_1825[['Date'] + [col for col in normalized_df_rows_1825.columns if col != 'Date']]
print(f"최종 데이터프레임 열 개수: {normalized_df_rows_1825.shape[1]}")
# Excel 파일로 저장
normalized_df_rows_1825.to_excel('data/output/rows_1825_with_normalize.xlsx', index=False)


# Step 4. 상관 분석
correlation_df = normalized_df_rows_1825.copy()
correlation_data_df = correlation_df.drop(columns=['Date']) # 상관분석을 위해 날짜(Date) 열 제거

print("데이터 타입과 결측치 확인: \n", correlation_data_df.info())  # 데이터 타입과 결측치 확인
print("데이터의 통계 요약 확인: \n", correlation_data_df.describe())  # 데이터의 통계 요약 확인

# 상관행렬 계산
correlation_matrix = correlation_data_df.corr()
# 상관행렬 히트맵 시각화
fig1, ax1 = plt.subplots(figsize=(20, 15)) # 첫 번째 그래프
sns.heatmap(correlation_matrix, cmap='coolwarm', ax=ax1, annot=False, fmt=".2f")
ax1.set_title("상관행렬", fontsize=16)
fig1.savefig("data/output/상관행렬_히트맵.png", dpi=300, bbox_inches='tight')


# Step 5-1 군집화 방식 (1) - 덴드로그램
# 거리 행렬 계산 (1 - 상관계수) : 덴드로그램(계층적 군집화)에만 사용
distance_matrix = 1 - correlation_matrix # 상관계수는 1에 가까울수록 유사함을 뜻한다,  (1−상관계수) 로 변환하여 거리를 계산한다.
print("거리행렬의 NaN 개수: ", distance_matrix.isna().sum().sum())  # NaN 값의 개수 확인
linked = linkage(distance_matrix, method='ward')
# 덴드로그램 시각화
fig2 = plt.figure(figsize=(15, 10))  # 두 번째 그래프
dendrogram(linked, labels=correlation_data_df.columns, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
plt.xlabel("Cryptocurrencies")
plt.ylabel("Distance")
fig2.savefig("data/output/덴드로그램_시각화.png", dpi=300, bbox_inches='tight')  # 두 번째 그래프 저장

# Step 5-2 군집화 방식 (2) - k-means 군집화
#  K 값 최적화를 위한 엘보우 방법
wcss = []
for i in range(2, 11):  # K 값을 2부터 10까지 시도
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(correlation_data_df)
    wcss.append(kmeans.inertia_)
# WCSS 값 시각화
fig3, ax3 = plt.subplots(figsize=(8, 5))  # fig3, ax3 생성
ax3.plot(range(2, 11), wcss, marker='o', linestyle='--')
ax3.set_title('Elbow Method for Optimal K', fontsize=16)
ax3.set_xlabel('Number of Clusters (K)', fontsize=12)
ax3.set_ylabel('WCSS', fontsize=12)
fig3.savefig("data/output/엘보우_방법.png", dpi=300, bbox_inches='tight')  # fig3로 저장
# elbow 는 K-Means 클러스터링의 적절한 군집 수(K)를 결정하기 위한 시각화이다.
# 기울기가 가장 완만해지는 지점이 적절한 군집 수 이다.


kmeans_k4 = KMeans(n_clusters=4, random_state=42)
cluster_labels_k4 = kmeans_k4.fit_predict(correlation_data_df)
kmeans_k5 = KMeans(n_clusters=5, random_state=42)
cluster_labels_k5 = kmeans_k5.fit_predict(correlation_data_df)

# 군집 레이블을 데이터프레임에 추가
k4_clustered_data_df = correlation_data_df.copy()
k4_clustered_data_df['Cluster'] = cluster_labels_k4
k5_clustered_data_df = correlation_data_df.copy()
k5_clustered_data_df['Cluster'] = cluster_labels_k5
# 군집 결과 저장
k4_clustered_data_df.to_csv("data/output/kmeans4_군집화_결과.csv", index=False)
k5_clustered_data_df.to_csv("data/output/kmeans5_군집화_결과.csv", index=False)




# python data_processing.py