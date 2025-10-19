import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau # 점차 학습률을 줄이는 용도
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F


# Step 0. 환경 체크
print("CUDA available:", torch.cuda.is_available()) # CUDA 사용 가능 여부 확인
print("Number of GPUs:", torch.cuda.device_count()) # 사용 가능한 디바이스 수
if torch.cuda.is_available(): # 현재 디바이스 확인
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# Step 1. 데이터 가져오기

file_path = 'data/output/코인데이터_onesheet_normalized.xlsx' # 정규화된 데이터
df = pd.read_excel(file_path)
df = df.drop(columns=['Date']) # Date 열 제외

# 예측할 열(종가)만 추출
# 'Close'가 포함된 열 중 'KRW'와 관련된 열 제외
target_columns = [col for col in df.columns if 'Close' in col and 'KRW' not in col]


# FX. 시계열 데이터 생성 함수
def create_sequences(data, target_data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])  # 모든 열 입력
        y.append(target_data[i + look_back])  # 종가 열만 예측
    # numpy array로 변환 후 torch tensor로 변환
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Fx. 데이터 분할 함수 정의
def data_split(X, y, method="expanding_window_split", validation_split=0.5, train_ratio=0.8, window_size=0.6,
               step_size=0.1):
    if method == "expanding_window_split":
        train_size = int(len(X) * train_ratio)
        val_size = int((len(X) - train_size) * validation_split)
        test_size = len(X) - train_size - val_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

    elif method == "sliding_window_validation":
        window_size = int(len(X) * window_size)
        step_size = int(len(X) * step_size)
        for start in range(0, len(X) - window_size, step_size):
            end = start + window_size
            X_train, y_train = X[start:end], y[start:end]
            X_val, y_val = X[end:end + step_size], y[end:end + step_size]
            X_test, y_test = X[end + step_size:], y[end + step_size:]

    elif method == "reverse_walk_forward_validation":
        train_size = int(len(X) * train_ratio)
        X_train = X[-train_size:]
        y_train = y[-train_size:]
        X_val = X[-(train_size + int((len(X) - train_size) * validation_split)):-train_size]
        y_val = y[-(train_size + int((len(X) - train_size) * validation_split)):-train_size]
        X_test = X[:-(train_size + len(X_val))]
        y_test = y[:-(train_size + len(y_val))]

    else:
        raise ValueError(
            "Invalid method. Choose from 'expanding_window_split', 'sliding_window_validation', or 'reverse_walk_forward_validation'.")

    return X_train, X_val, X_test, y_train, y_val, y_test



# Step 3. LSTM 모델 정의

# 단방향 LSTM 모델에 Attention 메커니즘을 추가한 클래스 정의
class UnidirectionalLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(UnidirectionalLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

        # Attention을 위한 추가 Layer 정의
        self.attention_layer = nn.Linear(hidden_size, 1)
        # hidden_size 크기를 1로 축소하여 각 시퀀스의 score를 계산하기 위함

    def attention(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attn_scores = self.attention_layer(lstm_output)  # (batch_size, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch_size, seq_len)로 차원 축소

        attn_weights = F.softmax(attn_scores, dim=1)  # Attention 확률값으로 변환
        # Attention 가중치를 사용하여 가중합 계산
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), lstm_output)  # (batch_size, 1, hidden_size)
        weighted_output = weighted_output.squeeze(1)  # (batch_size, hidden_size)

        return weighted_output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)

        # Attention 메커니즘 적용
        attn_output = self.attention(lstm_out)  # (batch_size, hidden_size)

        # Fully Connected Layer를 통과시켜 최종 출력값 생성
        output = self.fc(attn_output)  # (batch_size, output_size)

        return output



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


# Step 4. 학습 및 평가 (배치 처리 적용)
batch_size = 32  # 배치 크기 설정
results = {}
models = {}  # 각 타겟 열에 대해 모델을 저장
metrics_results = {}  # 평가 지표를 저장할 딕셔너리
look_back = 5

for target_col in target_columns:
    print(f"Training for {target_col}")

    # 시계열 데이터 생성
    X, y = create_sequences(df.values, df[target_col].values, look_back)

    # Step 4-1. 시계열 데이터 분리 (시간 순서 유지)
    train_size = int(len(X) * 0.9)
    test_val_size = len(X) - train_size

    # 검증과 테스트를 나누기 위한 비율 설정 (예: 검증 10%, 테스트 10%)
    validation_split = 0.5
    val_size = int(test_val_size * validation_split)
    test_size = test_val_size - val_size

    # 데이터 분할
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # TensorDataset 및 DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 설정
    input_size = df.shape[1]  # 모든 열 사용
    hidden_size = 256  # 은닉 유닛 수
    output_size = 1  # 종가 열 하나만 예측
    num_layers = 2  # LSTM 레이어 수
    learning_rate = 0.0002
    epochs = 30

    model = UnidirectionalLSTMWithAttention(input_size, hidden_size, output_size, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 학습 준비
    criterion = nn.MSELoss() # 평균 제곱 오차 : 연속형 값의 오차를 최소화하기 위한 일반적인 손실 함수.
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Adam의 가중치 감쇠(weight decay)를 개선한 방법

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early Stopping 조건 설정
    best_val_loss = float('inf')
    patience = 15
    trigger_times = 0

    # 학습 루프에서 배치 처리 적용
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 평균 학습 손실 계산
        avg_train_loss = running_loss / len(train_loader)

        # 검증 손실 계산
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss += criterion(y_val_pred.squeeze(-1), y_val_batch).item()

        avg_val_loss = val_loss / len(val_loader)

        # Early Stopping 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            print(f'Validation loss did not improve. Trigger times: {trigger_times}')

            if trigger_times >= patience:
                print("Early stopping triggered")
                break

        # 학습률 감소 스케줄러 업데이트
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # 평가 및 저장
    model.eval()
    y_test_np, y_pred_np = [], []
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_pred = model(X_test_batch)
            y_test_np.extend(y_test_batch.numpy())
            y_pred_np.extend(y_test_pred.cpu().numpy())

    y_test_np = np.array(y_test_np)  # 결과를 numpy로 변환
    y_pred_np = np.array(y_pred_np)  # 예측 결과도 numpy로 변환
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

    model = models[target_col]  # 해당 열에 대한 학습된 모델 가져오기
    last_sequence = df.values[-look_back:]  # 현재 데이터의 마지막 시퀀스를 가져옴

    # GPU로 데이터 이동
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    future_predictions = []  # 미래 예측을 저장할 리스트

    for step in range(future_steps):
        with torch.no_grad():
            # 현재 시퀀스로 예측 수행
            prediction = model(last_sequence)

            # 예측값을 시퀀스에 추가하여 다음 입력으로 사용
            next_value = prediction.squeeze(-1).cpu().numpy().item()  # 예측된 값을 numpy로 변환
            next_value = max(0, next_value)  # 0보다 작아지지 않도록 클리핑

            # 새로운 타임스텝을 생성하여 시퀀스에 추가
            # 기존의 마지막 타임스텝에서 종가를 제외한 나머지 피처를 유지하고, 종가 부분만 예측된 값으로 교체
            next_step = last_sequence[:, -1, :].cpu().numpy()  # 마지막 타임스텝을 가져옴
            next_step[:, -1] = next_value  # 마지막 피처(종가)를 예측값으로 대체

            # 새로운 시퀀스 구성 (가장 오래된 타임스텝 제거 후 새로운 타임스텝 추가)
            next_step_tensor = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(device)
            last_sequence = torch.cat([last_sequence[:, 1:, :], next_step_tensor], dim=1)

            # 예측값 저장
            future_predictions.append(next_value)

    # 정규화된 상태로 저장
    future_results[target_col] = future_predictions


# Step 6. 미래 예측 결과 시각화
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


# 모든 타겟 열에 대해 그래프 생성
for target_col in target_columns:
    print(f"Visualizing results for {target_col}")

    # Step 1: 학습 데이터 및 테스트 데이터에 대한 모델 예측
    y_train_pred = models[target_col](X[:train_size].clone().detach().to(device)).cpu().detach().numpy()
    y_test_pred = models[target_col](X[train_size:train_size + test_size].clone().detach().to(device)).cpu().detach().numpy()

    # Step 2: 미래 예측 날짜 생성
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')

    # Step 3: 예측 데이터 병합
    predicted_train_df = pd.DataFrame({
        'Date': df.index[:train_size],
        'Prediction': y_train_pred.flatten()
    }).set_index('Date')

    predicted_test_df = pd.DataFrame({
        'Date': df.index[train_size:train_size + test_size],
        'Prediction': y_test_pred.flatten()
    }).set_index('Date')

    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Prediction': future_results[target_col]
    }).set_index('Date')

    # Step 4: 기존 Close 값과 병합
    combined_df = pd.concat([
        df[[target_col]].rename(columns={target_col: 'Actual'}),
        predicted_train_df,
        predicted_test_df,
        future_predictions_df
    ], axis=0)

    # Step 5: 그래프에 추가 (알파값 조정 및 색상 매핑)
    color = colors.get(target_col, "black")  # 지정된 색상 없으면 기본 검정색
    plt.plot(combined_df.index, combined_df['Actual'], label=f'Actual {target_col}', color=color, alpha=0.4)
    plt.plot(combined_df.index, combined_df['Prediction'], linestyle='dotted', label=f'Predicted {target_col}', color=color, alpha=0.9)

# 그래프 설정
plt.legend()
plt.title('LSTM Prediction with Future Forecast for All Targets')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.grid(True)
plt.show()
