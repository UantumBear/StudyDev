import pandas as pd
""" 조별과제에 쓴 factorloading_OHLC 전처리 """

# 엑셀 파일 읽기
file_path = 'data/factorloading_OHLC.xlsx'
xls_ori = pd.ExcelFile(file_path)


# Fx 1. 날짜형으로 변환
def ensure_datetime(dataframe, date_col_name:str="Date"):
    """ Date 열을 datetime으로 변환하여 반환하는 함수 """
    dataframe[date_col_name] = pd.to_datetime(dataframe[date_col_name], errors='coerce')  # Date 열을 datetime 형식으로 변환
    return dataframe

# Fx 2. 숫자형으로 변환
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

# Fx 3. 시트들을 {시트명}_{열명} 으로 모아 하나의 시트로 병합
def merge_sheets(xlsx_path):
    """
        여러 시트를 {시트명}_{열명} 형식으로 병합하는 함수.
        Args:
            file_path (str): Excel 파일 경로
        Returns:
            하나의 최종 sheet 를 가진 xlsx 파일 저장 후,
            pd.DataFrame: 병합된 DataFrame 반환
        """

    # Excel 파일 불러오기
    excel_file = pd.ExcelFile(xlsx_path)
    sheet_names = excel_file.sheet_names  # 모든 시트 이름 가져오기

    merged_df = pd.DataFrame()  # 빈 DataFrame 생성
    # 각 시트 데이터를 처리하여 병합
    for sheet_name in sheet_names:
        # 시트 데이터 읽기
        df = excel_file.parse(sheet_name)
        # 열 이름에 시트 이름 추가 (예: "LUNC-USD_Open", "LUNC-USD_High")
        df = df.rename(columns={col: f"{sheet_name}_{col}" for col in df.columns if col != 'Date'})

        # 병합 (날짜를 기준으로)
        if merged_df.empty:
            merged_df = df  # 첫 번째 DataFrame은 그대로 사용
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')  # 'Date' 기준 병합
    # 병합 결과 출력
    print(f"병합 결과: {merged_df.shape[0]} 행, {merged_df.shape[1]} 열")
    print(f"merged_df.head():\n{merged_df.head()}")
    print(f"merged_df.info():\n{merged_df.info()}")

    # 병합된 결과를 Excel 파일로 저장
    merged_df.to_excel("data/output/02_하나의시트로통합한엑셀.xlsx", index=False)
    print(f"병합된 결과가 'data/output/'에 저장되었습니다.")

    return merged_df

# Fx 4. 특정 열들을 선형보간처리하여 엑셀을 저장하고 dataframe 으로 반환하는 함수
def process_null(dataframe, columns, date_col='Date', output_file=None):
    """
    특정 열에 대해 선형 보간 처리 및 결과 저장.

    Args:
        dataframe (pd.DataFrame): 원본 데이터 프레임.
        columns (list): 처리할 열 이름 리스트.
        date_col (str): 기준 날짜 열 이름 (기본값: 'Date').
        output_file (str): 처리 결과를 저장할 파일 경로 (지정하지 않으면 저장하지 않음).

    Returns:
        pd.DataFrame: 선형 보간 처리가 완료된 데이터 프레임.
    """
    # 날짜 열을 datetime 형식으로 변환
    dataframe[date_col] = pd.to_datetime(dataframe[date_col])

    # 1일 간격으로 날짜 범위 생성
    date_range = pd.date_range(start=dataframe[date_col].min(), end=dataframe[date_col].max(), freq='D')
    df_full = pd.DataFrame({date_col: date_range})

    # 날짜 기준 병합
    merged_df = pd.merge(df_full, dataframe, on=date_col, how='left')

    # 처리할 열에 대해 선형 보간 수행
    for col in columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].interpolate(method='linear')
        else:
            print(f"열 '{col}'이(가) 데이터 프레임에 없습니다. 건너뜁니다.")

    # 결과 저장
    if output_file:
        merged_df.to_excel(output_file, index=False)
        print(f"처리 결과가 '{output_file}'에 저장되었습니다.")

    return merged_df


# Step 0. 엑셀 파일 읽고 dataframe 에 저장

data_df = merge_sheets(file_path)


# Step 1. 환율 데이터 전처리 - null 값을 선형 보간 처리
# xls = pd.ExcelFile("data/output/02_하나의시트로통합한엑셀.xlsx")
# 모든 시트를 읽어서 딕셔너리로 반환
all_sheets = pd.read_excel("data/output/02_하나의시트로통합한엑셀.xlsx", sheet_name=None)
all_df = all_sheets["Sheet1"]  # "Sheet1" 데이터프레임 가져오기
# 데이터프레임 열 이름 확인
print(f"all_df.columns: \n{all_df.columns}")
all_not_KRW_Volume_df = all_df.drop("KRW-Exchange-Rate_Volume", axis=1) # 열을 삭제
print(all_not_KRW_Volume_df.head())

target_cols = ["KRW-Exchange-Rate_Open","KRW-Exchange-Rate_Close","KRW-Exchange-Rate_Low","KRW-Exchange-Rate_High"]
processed_df = process_null(
    dataframe=all_not_KRW_Volume_df,
    columns=target_cols,
    date_col="Date",
    output_file="data/output/02_하나의시트로통합후_널값을보간처리한엑셀.xlsx"
)

processed_not_null_df = processed_df.dropna()
processed_not_null_df.to_excel("data/output/02_하나의시트로통합후_널값을보간처리후_남은널행삭제.xlsx", index=False)
