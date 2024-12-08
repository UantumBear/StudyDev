from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Step 1. LUNC
# Train 데이터와 Validation 데이터 로드
train_path = 'data/output/02_cluster_0_for_train.xlsx'
train_data = pd.read_excel(train_path)

# Date 열 제외한 정규화 대상 열
feature_columns = train_data.columns.difference(['Date'])

# Train 데이터 정규화
scaler = MinMaxScaler()
train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])

# Validation 데이터 정규화
target_scaler = MinMaxScaler()

# 필요한 경우 파일로 저장
train_data.to_excel('data/output/02_cluster_0_normalized_train_data.xlsx', index=False)


# Step 2. DASH
# Train 데이터와 Validation 데이터 로드
train_1_path = 'data/output/02_cluster_1_for_train.xlsx'
train_1_data = pd.read_excel(train_1_path)

# Date 열 제외한 정규화 대상 열
feature_1_columns = train_1_data.columns.difference(['Date'])

# Train 데이터 정규화
scaler_1 = MinMaxScaler()
train_1_data[feature_1_columns] = scaler_1.fit_transform(train_1_data[feature_1_columns])

# Validation 데이터 정규화
target_1_scaler = MinMaxScaler()

# 필요한 경우 파일로 저장
train_1_data.to_excel('data/output/02_cluster_1_normalized_train_data.xlsx', index=False)

# Step 2. DASH
# Train 데이터와 Validation 데이터 로드
train_2_path = 'data/output/02_cluster_2_for_train.xlsx'
train_2_data = pd.read_excel(train_2_path)

# Date 열 제외한 정규화 대상 열
feature_2_columns = train_2_data.columns.difference(['Date'])

# Train 데이터 정규화
scaler_2 = MinMaxScaler()
train_2_data[feature_2_columns] = scaler_2.fit_transform(train_2_data[feature_2_columns])

# Validation 데이터 정규화
target_2_scaler = MinMaxScaler()

# 필요한 경우 파일로 저장
train_2_data.to_excel('data/output/02_cluster_2_normalized_train_data.xlsx', index=False)