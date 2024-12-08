import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 0. 환경 체크
# CUDA 사용 가능 여부 확인
print("CUDA available:", torch.cuda.is_available())
# 사용 가능한 디바이스 수
print("Number of GPUs:", torch.cuda.device_count())
# 현재 디바이스
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# Step 1. 데이터 가져오기

file_path = 'data/output/코인데이터_onesheet_normalized.xlsx' # 정규화된 데이터
df = pd.read_excel(file_path)
df = df.drop(columns=['Date']) # Date 열 제외

# 예측할 열(종가)만 추출
# 'Close'가 포함된 열 중 'KRW'와 관련된 열 제외
target_columns = [col for col in df.columns if 'Close' in col and 'KRW' not in col]


# Step 2. 시계열 데이터 생성 함수
# def create_sequences(data, target_data, look_back=30):
#     X, y = [], []
#     for i in range(len(data) - look_back):
#         X.append(data[i:i + look_back])  # 모든 열 입력
#         y.append(target_data[i + look_back])  # 종가 열만 예측
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
def create_sequences(data, target_data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])  # 모든 열 입력
        y.append(target_data[i + look_back])  # 종가 열만 예측

    # numpy array로 변환 후 torch tensor로 변환
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

look_back = 50


# Step 3. LSTM 모델 정의
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # 마지막 타임스텝 출력
#         return out
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # 출력 값을 0~1로 제한
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝 출력
        return out

# Step 4. 학습 및 평가

# Fx. 평가 함수
def evaluate_model(y_true, y_pred):
    mask = y_true != 0
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    mse = mean_squared_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true[mask], y_pred[mask])

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}


# Step 4. 학습 및 평가
results = {}
models = {}  # 각 타겟 열에 대해 모델을 저장
metrics_results = {}  # 평가 지표를 저장할 딕셔너리

for target_col in target_columns:
    print(f"Training for {target_col}")

    # 시계열 데이터 생성
    X, y = create_sequences(df.values, df[target_col].values, look_back)

    # Step 3. 시계열 데이터 분리 (시간 순서 유지)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    test_size = len(X) - train_size - val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # 확인
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")


    # 모델 설정
    input_size = df.shape[1]  # 모든 열 사용
    hidden_size = 128
    output_size = 1  # 종가 열 하나만 예측
    num_layers = 3
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 학습 준비
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 학습 루프
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred.squeeze(-1), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    # 평가 및 저장
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # 결과 저장
    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    results[target_col] = {'y_test': y_test_np, 'y_pred': y_pred_np}
    models[target_col] = model  # 모델 저장

    # 평가 함수 호출 및 출력
    print(f"\nEvaluation for {target_col}:")
    metrics = evaluate_model(y_test_np, y_pred_np)
    metrics_results[target_col] = metrics  # 평가 지표 저장



# Step 5. 미래 예측
future_steps = 365  # 예측할 미래의 일 수
future_results = {}

for target_col in target_columns:
    print(f"Future forecasting for {target_col}")

    model = models[target_col]  # 해당 열에 대한 모델 가져오기
    last_sequence = df.values[-look_back:]  # 현재 데이터의 마지막 시퀀스를 가져옴

    # GPU로 데이터 이동
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"last_sequence shape: {last_sequence[:, 1:, :].shape}")


    future_predictions = []  # 미래 예측을 저장할 리스트

    for _ in range(future_steps):
        with torch.no_grad():
            # 현재 시퀀스로 예측 수행
            prediction = model(last_sequence)
            prediction_value = prediction.squeeze(-1).cpu().numpy().item()  # 스칼라 값 저장
            future_predictions.append(prediction_value)

            # 예측된 값을 input_size에 맞게 확장
            pred_expanded = prediction.unsqueeze(-1).expand(-1, 1, last_sequence.shape[2])  # (batch_size, 1, input_size)
            print(f"pred_expanded shape: {pred_expanded.shape}")
            # 시퀀스 업데이트
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred_expanded), dim=1)  # seq_len 업데이트




    # 정규화된 상태로 저장
    future_results[target_col] = future_predictions





# Step 7. 미래 예측 결과 시각화
plt.figure(figsize=(15, 8))

# Date 열 포함하여 데이터 로드
df = pd.read_excel(file_path)
df.set_index('Date', inplace=True)

# 색상 매핑 (고정된 5가지 색상)
colors = {
    "Close BTC-USD": "blue",
    "Close MTL-USD": "orange",
    "Close USDT-USD": "green",
    "Close GT-USD": "red",
    "Close FLUX-USD": "purple"
}

# 타겟 열별로 예측 결과를 시각화
for target_col in target_columns:
    print(f"Visualizing results for {target_col}")

    # Step 7-1: 테스트 데이터 예측
    X_target = create_sequences(df.values, df[target_col].values, look_back)[0]
    test_size = X_test.shape[0]
    y_test_pred = model(X_target[-test_size:].cuda()).cpu().detach().numpy()

    # Step 7-2: 미래 예측 날짜 생성
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')

    # Step 7-3: 데이터프레임 구성
    actual_df = df[[target_col]].rename(columns={target_col: "Actual"})
    predicted_test_df = pd.DataFrame(
        {"Date": df.index[-test_size:], "Prediction": y_test_pred.flatten()}
    ).set_index("Date")
    future_predictions_df = pd.DataFrame(
        {"Date": future_dates, "Prediction": future_predictions}
    ).set_index("Date")

    # Step 7-4: 기존 데이터와 예측 결과 병합
    combined_df = pd.concat([actual_df, predicted_test_df, future_predictions_df], axis=0)

    # Step 7-5: 그래프 생성
    color = colors.get(target_col, "black")  # 색상 매핑, 기본 검정색
    plt.plot(combined_df.index, combined_df["Actual"], label=f"Actual {target_col}", color=color, alpha=0.4)
    plt.plot(combined_df.index, combined_df["Prediction"], linestyle="dotted", label=f"Predicted {target_col}", color=color, alpha=0.9)

# 그래프 설정
plt.legend()
plt.title("LSTM Prediction with Future Forecast for All Targets")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.grid(True)
plt.show()
