import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

### 함수 정의
def check_missing_rows(data, dataname:str="UnKnown"):
    """ 01 기본 데이터 확인 """
    # print(f"\033[91m{dataname}.index: \033[0m\n{data.index}")
    print(f"\033[91mtype({dataname}): \033[0m\n{type(data)}")

    pd.set_option('display.max_columns', None) # 모든 열이 표시되도록 설정
    print(f"\033[91m{dataname}.head(): \033[0m\n{data.head()}")
    pd.reset_option('display.max_columns') # 설정 원래대로 되돌리기

    print(f"\033[91m{dataname}: \033[0m\n{data}")

    """ 02 DataFrame 에서 빈 행이 있는지 체크 """
    # 빈 값이 있는 행 확인
    missing_rows = data[data.isnull().any(axis=1)]
    if not missing_rows.empty:
        print(f"\033[91m{dataname}: \033[0m빈 값이 있는 행이 있습니다:")
        print(missing_rows)
    else:
        print(f"\033[91m{dataname}: \033[0m빈 값이 있는 행이 없습니다.")



# Step 0-1. matplotlib 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Step 0-2. 날짜 설정
# 다운로드 받을 데이터의 날짜 범위
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)
end_date_str = end_date.strftime("%Y-%m-%d")
start_date_str = start_date.strftime("%Y-%m-%d")
# plot 을 구성할 날짜 범위
plot_end_date = datetime.today() + timedelta(days=365 * 2)
plot_start_date = end_date - timedelta(days=365 * 10)

# Step 0. 주가 데이터 다운로드
ticker = '055550.KS' # 신한금융지주
data_ori = yf.download(ticker, start=start_date_str, end=end_date_str)
data_ori.to_csv('data_ori.csv', index=True) # 다른 경로에 CSV 파일 저장 - 눈으로 데이터 확인하는 용도 (거래일 데이터만 있음)

data = data_ori.copy() # 원본 데이터 보존을 위해 데이터 복사

# Step 1. 데이터 전처리
# Step 1-1. 데이터 확인 및 Pandas 재구성
check_missing_rows(data, dataname="data") # 빈 값이 있는 행이 있는지 확인
# data = data.reset_index() # 멀티 인덱스 해제
data.columns = data.columns.droplevel('Ticker') # 'Ticker' 레벨을 제거하고 'Date'만 인덱스로 남김

close_data_pre = data[['Close']] # 'Close' 열 선택
close_data_pre.index = pd.DatetimeIndex(data.index.date) # freq='B') # 비즈니스 데이터로 하면 완벽히 맞아 떨어지질 않음.. 공휴일 때문인지

# Step 1-2. 결측치 채우기 - 일일날짜 계산
# 비즈니스 데이가 완벽하게 읽히지 않는 것 같아서, 일단은 결측치를 이전 거래일 기준 값으로 채워넣음.
full_date_range = pd.date_range(start=close_data_pre.index.min(), end=close_data_pre.index.max(), freq='D')
close_data_full = close_data_pre.reindex(full_date_range).fillna(method='ffill') # 데이터를 전체 날짜 범위로 리샘플링 후 빈 값 채우기

# Step 1-3. 다시 비즈니스 데이로 변경
# 'B' 주기를 사용하여 주말을 제외한 비즈니스 날짜만 남김
business_date_range = pd.date_range(start=close_data_full.index.min(), end=close_data_full.index.max(), freq='B')
close_data = close_data_full.reindex(business_date_range).fillna(method='ffill')
print(close_data.head(15))  # 상위 15개 데이터 출력하여 확인


# 통계 요약으로 이상치 확인
print("통계 요약:")
print(close_data.describe())


# close_data_copy = data[['Close']]
adj_close_data = data[['Adj Close']]
data['5MA'] = data['Adj Close'].rolling(window=5).mean() # 5일 이동 평균 열 추가
data['20MA'] = data['Adj Close'].rolling(window=20).mean() # 20일 이동 평균 열 추가
data['60MA'] = data['Adj Close'].rolling(window=60).mean() # 5일 이동 평균 열 추가
data['120MA'] = data['Adj Close'].rolling(window=120).mean() # 20일 이동 평균 열 추가
ma5_data = data['5MA']
ma20_data = data['20MA']
ma60_data = data['60MA']
ma120_data = data['120MA']



check_missing_rows(close_data, dataname="close_data") # 빈 값이 있는 행이 있는지 확인
# Step 1-2. 정규화
# 1. 로그 변환
log_close_data = np.log(close_data['Close'].fillna(method='ffill'))



# Step 2. 최적의 ARIMA 모델 찾기
# p, d, q 값의 범위 설정
p_values = range(0, 4)  # p 값 범위를 늘려서 테스트
d_values = range(0, 2)  # 차분(d) 범위를 0~2까지
q_values = range(0, 4)  # q 값 범위를 늘려서 테스트

best_aic = np.inf
best_order = None
best_model_aic = None

# 최적의 (p, d, q) 조합을 찾기
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # model = ARIMA(log_close_data, order=(p, d, q))
                model = ARIMA(close_data, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                print(f"ARIMA({p},{d},{q}) - AIC: {aic}")

                # AIC가 가장 낮은 모델을 업데이트
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model_aic = fitted_model

            except Exception as e:
                print(f"ARIMA({p},{d},{q}) - Error: {e}")

# 최적의 모델 정보 출력
if best_model_aic is None:
    print("최적의 모델을 찾지 못했습니다.")
else:
    print(f'\nBest AIC Model: ARIMA{best_order} - AIC: {best_aic}')
    print(f'\nBest AIC Model: {best_model_aic.model_orders}')
    print(f"best_model_aic.summary(): ")
    print(best_model_aic.summary())



# Step 3. 최적의 모델로 향후 1년간 종가 예측
# 백테스트 기간 설정 (예: 과거 10년 동안)
backtest_period = len(close_data) # 252 * 10 이렇게 쓰면는  과거 10년의 주중 거래일 수 기준 (맞지않음, 공휴일도 있어서 그런듯)
# 과거 구간 예측 수행 (백테스트)
backtest_object = best_model_aic.get_forecast(steps=backtest_period)
backtest_log_scale = backtest_object.predicted_mean
backtest_result = np.exp(backtest_log_scale)  # 원래 스케일로 복원
# 신뢰 구간을 추출하고 원래 스케일로 복원
backtest_conf_int_log_scale = backtest_object.conf_int()
backtest_conf_int = np.exp(backtest_conf_int_log_scale)

# 백테스트 날짜 인덱스 생성
backtest_dates = close_data.index[-backtest_period:]
# 인덱스에 빈도 설정
if not backtest_dates.inferred_freq:
    backtest_dates = pd.PeriodIndex(backtest_dates)
#backtest_df = pd.DataFrame(backtest_result.values, index=backtest_dates, columns=['Backtest Forecast'])
backtest_df = pd.DataFrame({'Backtest Forecast': backtest_log_scale}, index=backtest_dates)
backtest_conf_int_df = pd.DataFrame(backtest_conf_int.values, index=backtest_dates, columns=['Lower Bound', 'Upper Bound'])

# 향후 1년간 예측 (미래 예측)
forecast_period = 252  # 향후 1년 예측
forecast_object = best_model_aic.get_forecast(steps=forecast_period)
forecast_log_scale = forecast_object.predicted_mean
forecast_result = np.exp(forecast_log_scale)

# 미래 예측 신뢰 구간을 추출하고 원래 스케일로 복원
forecast_conf_int_log_scale = forecast_object.conf_int()
forecast_conf_int = np.exp(forecast_conf_int_log_scale)

# 미래 예측 날짜 인덱스 생성
last_date = close_data.index[-1]
forecast_dates = pd.date_range(last_date, periods=forecast_period + 1, freq='B')[1:]
#forecast_df = pd.DataFrame(forecast_result.values, index=forecast_dates, columns=['Forecast'])
forecast_df = pd.DataFrame({'Forecast': forecast_log_scale}, index=forecast_dates)
forecast_conf_int_df = pd.DataFrame(forecast_conf_int.values, index=forecast_dates, columns=['Lower Bound', 'Upper Bound'])

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
# 실제 데이터
plt.plot(close_data, label="Actual Close Price")
plt.plot(adj_close_data, label="Adj Close Price", color="black")
plt.plot(ma5_data, label="5MA Price", color="orange")
plt.plot(ma20_data, label="20MA Price", color="green")
plt.plot(ma60_data, label="60MA Price", color="red")
plt.plot(ma120_data, label="120MA Price", color="purple")

# 예측 데이터
plt.plot(backtest_df, label="Backtest Forecast (10 years)", linestyle='--', color='green')
# plt.fill_between(backtest_df.index, backtest_conf_int_df['Lower Bound'], backtest_conf_int_df['Upper Bound'], color='green', alpha=0.2)
plt.fill_between(backtest_df.index, backtest_conf_int_df['Lower Bound'], backtest_conf_int_df['Upper Bound'], color='green', alpha=0.2)
plt.plot(forecast_df, label="1-Year Forecast", linestyle='--', color='orange')
# plt.fill_between(forecast_df.index, forecast_conf_int_df['Lower Bound'], forecast_conf_int_df['Upper Bound'], color='pink', alpha=0.2)
plt.fill_between(forecast_df.index, forecast_conf_int_df['Lower Bound'], forecast_conf_int_df['Upper Bound'], color='pink', alpha=0.2)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("10-Year Backtest and 1-Year Price Forecast with Confidence Interval")
plt.legend()
plt.show()

