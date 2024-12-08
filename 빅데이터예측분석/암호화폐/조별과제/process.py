import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Fx 1. 열이름 병합
def rename_column(dataframe, basis_col_name:str):
    """ Ticker 를 포함한 여러 행으로 구성된 열 이름을 단일 열 이름으로 병합하여 반환하는 함수 """
    df = dataframe.copy()

    # 기준 열의 행 위치 구하기
    basis_row_idx = df[df.iloc[:,0] == basis_col_name].index[0]
    print(f"basis_col_name: {basis_col_name}, basis_row_idx: {basis_row_idx}")

    # 열 이름 생성: 'basis_col_name' 위의 행들 병합
    headers = df.iloc[:basis_row_idx].fillna('').astype(str)
    merged_headers = headers.apply(lambda x: ' '.join(x).strip(), axis=0)
    df.columns = merged_headers
    df.columns.values[0] = basis_col_name  # 첫 번째 열 이름을 'basis_col_name'로 명시적으로 지정

    # 실제 데이터만 남기기 ('Date'가 있는 행 바로 아래부터)
    df = df.iloc[basis_row_idx + 1:].reset_index(drop=True)
    return df

# Fx 2. 날짜형으로 변환
def ensure_datetime(dataframe, date_col_name:str="Date"):
    """ Date 열을 datetime으로 변환하여 반환하는 함수 """
    dataframe[date_col_name] = pd.to_datetime(dataframe[date_col_name], errors='coerce')  # Date 열을 datetime 형식으로 변환
    return dataframe

# Fx 3. 숫자형으로 변환
def ensure_numeric(dataframe, except_col_name:str, func:str="zero"):
    numeric_columns = dataframe.columns.difference([except_col_name])  # Date 열 제외
    for col in numeric_columns:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')  # 숫자로 변환, 변환 불가한 값은 NaN 처리
    # 숫자형 데이터 결측값 처리
    if func == "linear interpolation":
        # Nan 값을 선형보간
        dataframe[numeric_columns] = dataframe[numeric_columns].interpolate(method='linear')
    elif func == "zero":
        # NaN 값을 0으로 채움
        dataframe[numeric_columns] = dataframe[numeric_columns].fillna(0)

    return dataframe


# Step 1. 환율 데이터 전처리 - null 값을 선형 보간 처리
# 엑셀 파일 읽기
file_path = 'data/yfinance_results_환율추가버전.xlsx'
xls = pd.ExcelFile(file_path)

# 'KRW-Exchange-Rate' 시트 읽기
KRW_ori_df = pd.read_excel(xls, sheet_name='KRW-Exchange-Rate', header=None)
print("KRW_ori_df.head() : \n" , KRW_ori_df.head()) # 데이터 확인용

# Fx 1. 단일 열 이름으로 병합하기
KRW_df = rename_column(dataframe=KRW_ori_df, basis_col_name="Date")

# Date 열을 datetime 형식으로 변환
KRW_df['Date'] = pd.to_datetime(KRW_df['Date'])
print("셀 병합 처리 - KRW_df.head() : \n" , KRW_df.head()) # 데이터 확인용

# 1일 간격으로 날짜 범위 생성
date_range = pd.date_range(start=KRW_df['Date'].min(), end=KRW_df['Date'].max(), freq='D')

# 원본 데이터에 1일 간격 데이터가 없는 날짜를 NaN으로 채우기 위해 병합
df_full = pd.DataFrame(date_range, columns=['Date'])
krw_df = pd.merge(df_full, KRW_df, on='Date', how='left')
print("pandas 데이터 타입 확인 : \n " , krw_df.dtypes) # 데이터 확인용

# 숫자형 변환 (NaN 처리 포함)
krw_df = ensure_numeric(dataframe=krw_df, except_col_name="Date", func="linear interpolation")

# 결과를 Excel 파일로 저장
output_file = 'data/output/null값을_보간처리한_환율데이터.xlsx'
krw_df.to_excel(output_file, index=False)

# Step 2. 보간 처리 필요 없는 나머지 시트 읽기
BTC_ori_df = pd.read_excel(xls, sheet_name='BTC-USD', header=None)
btc_df = rename_column(dataframe=BTC_ori_df, basis_col_name="Date") # 단일 행으로 병합
btc_df = ensure_datetime(dataframe=btc_df, date_col_name="Date")

MTL_ori_df = pd.read_excel(xls, sheet_name='MTL-USD', header=None)
mtl_df = rename_column(dataframe=MTL_ori_df, basis_col_name="Date") # 단일 행으로 병합
mtl_df = ensure_datetime(dataframe=mtl_df, date_col_name="Date")

USDT_ori_df = pd.read_excel(xls, sheet_name='USDT-USD', header=None)
usdt_df = rename_column(dataframe=USDT_ori_df, basis_col_name="Date") # 단일 행으로 병합
usdt_df = ensure_datetime(dataframe=usdt_df, date_col_name="Date")

GT_ori_df = pd.read_excel(xls, sheet_name='GT-USD', header=None)
gt_df = rename_column(dataframe=GT_ori_df, basis_col_name="Date") # 단일 행으로 병합
gt_df = ensure_datetime(dataframe=gt_df, date_col_name="Date")

FLUX_ori_df = pd.read_excel(xls, sheet_name='FLUX-USD', header=None)
flux_df = rename_column(dataframe=FLUX_ori_df, basis_col_name="Date") # 단일 행으로 병합
flux_df = ensure_datetime(dataframe=flux_df, date_col_name="Date")

# df 들을 한데 모아 저장할 엑셀 파일 경로 설정
output_file = "data/output/코인데이터_sheets.xlsx"

# 여러 시트를 가진 엑셀 파일 생성
with pd.ExcelWriter(output_file) as writer:
    # 각각의 데이터프레임을 시트로 저장
    btc_df.to_excel(writer, sheet_name="BTC-USD", index=False)
    mtl_df.to_excel(writer, sheet_name="MTL-USD", index=False)
    usdt_df.to_excel(writer, sheet_name="USDT-USD", index=False)
    gt_df.to_excel(writer, sheet_name="GT-USD", index=False)
    flux_df.to_excel(writer, sheet_name="FLUX-USD", index=False)
    krw_df.to_excel(writer, sheet_name="KRW-Exchange-Rate", index=False)

print(f"데이터가 {output_file}에 저장되었습니다.")
print(f"usdt_df: \n {usdt_df}")
# 데이터 병합
merged_df = btc_df  # 기준 데이터프레임 설정
for df in [mtl_df, usdt_df, gt_df, flux_df, krw_df]:
    merged_df = pd.merge(merged_df, df, on='Date', how='outer')  # Date 열 기준 병합

# 최종적으로 null 값 drop 하기  (환율 데이터가 29일까지, 코인 데이터가 30일까지 있음)
merged_df = merged_df.dropna()

# 결과 확인
print(f"merged_df.head():\n ", merged_df.head())

# 엑셀 파일로 저장
output_file = "data/output/코인데이터_onesheet.xlsx"
merged_df.to_excel(output_file, index=False)
print(f"데이터가 {output_file}에 저장되었습니다.")

# Step 2. 모든 입력 변수들에 대해 정규화 수행
target_df = merged_df.copy()
# Date 열 제외
target_numeric_columns = target_df.columns.difference(['Date'])

# MinMaxScaler를 사용한 정규화
scaler = MinMaxScaler()
target_df[target_numeric_columns] = scaler.fit_transform(target_df[target_numeric_columns])



# 정규화된 데이터 저장
output_file = 'data/output/코인데이터_onesheet_normalized.xlsx'
target_df.to_excel(output_file, index=False)

print(f"정규화된 데이터가 {output_file}에 저장되었습니다.")