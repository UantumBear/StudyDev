import pandas as pd




from statsmodels.tsa.stattools import adfuller

def random_walk_check(xlsx_file_path, col_name):
    #file_path = 'data/bsv.xlsx'  # 파일 경로

    data = pd.ExcelFile(xlsx_file_path)  # 엑셀 파일 읽기
    data.sheet_names  # 시트 이름 확인
    df = data.parse('Sheet1')  # Sheet1 데이터를 읽어와 랜덤워크 여부를 확인
    df.head()  # 데이터 프레임의 첫 몇 행 확인

    # 'BSV-USD' 컬럼을 대상으로 랜덤워크 여부를 확인
    price_data = df[col_name]

    # 단위근 검정 (ADF Test)
    adf_result = adfuller(price_data)

    # 결과 출력
    adf_stat, p_value = adf_result[0], adf_result[1]
    result_print = f"""
    "ADF Statistic": {adf_stat},
    "p-value": {p_value},
    # p-value가 0.05보다 크면 랜덤워크일 가능성이 높음
    """
    print(result_print)

random_walk_check('data/bsv.xlsx', 'BSV-USD')
random_walk_check('data/output/로그변환후정규화한BSV.xlsx', 'BSV-USD_Close')

"""
[1] ADF Statistic (Augmented Dickey-Fuller Statistic) 란?
시계열 데이터가 '단위근' 을 가지고 있는지, 즉, 
데이터가 비정상적이고, 랜덤워크인지  를 검정하기 위해 사용되는 통계지표이다.
ADF Statistic 가 낮을수록 단위근이 없고 데이터가 정상적(stationary)일 가능성이 크다.

[2] p-value 란?
유의확률, 이것은 귀무가설(null hypothesis)가 옳은지 확률을 나타낸다.
ADF 테스트의 귀무가설은 "데이터가 랜덤워크이다" 이고,
즉 p-value 가 높을 수록, 데이터가 랜덤워크일 가능성이 높음을 의미한다.
보통 기준은 0.05로 한다.
"""