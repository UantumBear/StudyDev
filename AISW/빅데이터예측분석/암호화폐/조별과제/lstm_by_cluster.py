import pandas as pd

#업로드된 시계열 데이터 파일 확인
ohlc_data_path = 'data/factorloading_OHLC.xlsx'
ohlc_data = pd.read_excel(ohlc_data_path)

# 데이터 내용 및 구조 확인
ohlc_data.head(), ohlc_data.info()