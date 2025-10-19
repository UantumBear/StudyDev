import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt
import json # 최적화 파라미터를 찾은 결과를 저장해두기 위함 (optuna)
import os

# Step 0. JSON 파일 경로 설정
json_file_path = "data/output/best_hyperparameters_by_optuna.json"

# Step 1. 환경 체크
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Step 2. 데이터 가져오기
file_path = 'data/output/코인데이터_onesheet_normalized.xlsx'
df = pd.read_excel(file_path)
df.set_index('Date', inplace=True)  # Date를 인덱스로 설정

# 예측할 열(종가)만 추출
target_columns = [col for col in df.columns if 'Close' in col and 'USD' in col]

# FX. 시계열 데이터 생성 함수
def create_sequences(data, target_data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(target_data[i + look_back])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

# 시계열 데이터 분할
look_back = 5
X, y = create_sequences(df.values, df[target_columns[0]].values, look_back)

train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Step 3. LSTM 모델 정의
class UnidirectionalLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(UnidirectionalLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention_layer = nn.Linear(hidden_size, 1)

    def attention(self, lstm_output):
        attn_scores = self.attention_layer(lstm_output).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted_output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_output = self.attention(lstm_out)
        output = self.fc(attn_output)
        return output

# Step 4. Optuna로 하이퍼파라미터 최적화
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = UnidirectionalLSTMWithAttention(input_size=df.shape[1], hidden_size=hidden_size, output_size=1, num_layers=num_layers)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    for epoch in range(10):  # 하이퍼파라미터 탐색 시 Epoch은 적게
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.cuda(), y_val_batch.cuda()
                y_val_pred = model(X_val_batch)
                val_loss += criterion(y_val_pred.squeeze(-1), y_val_batch).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

# Step 4. JSON 파일 읽기 또는 Optuna 최적화
if os.path.exists(json_file_path):
    # JSON 파일이 있으면 읽기
    print("Loading hyperparameters from JSON file...")
    with open(json_file_path, "r") as f:
        best_params = json.load(f)  # JSON에서 하이퍼파라미터 불러오기
else:
    # JSON 파일이 없으면 Optuna로 최적화 수행
    print("JSON file not found. Running Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # 최적 하이퍼파라미터 저장
    best_params = study.best_params
    with open(json_file_path, "w") as f:
        json.dump(best_params, f)
    print("Optimization complete. Hyperparameters saved to JSON file.")

# 최적의 하이퍼파라미터 출력
print("Best hyperparameters:", best_params)

# Step 5. 코인별 독립적인 모델 학습 및 예측
future_steps = 365  # 미래 예측 일 수
future_results = {}
models = {}

for target_col in target_columns:
    print(f"Training and forecasting for {target_col}...")

    # 코인별 데이터 분리
    X, y = create_sequences(df.values, df[target_col].values, look_back)
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    test_size = len(X) - train_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # 모델 초기화
    model = UnidirectionalLSTMWithAttention(
        input_size=df.shape[1],
        hidden_size=best_params['hidden_size'],
        output_size=1,
        num_layers=best_params['num_layers']
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 생성
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=best_params['batch_size'], shuffle=False)

    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])

    # 학습 루프
    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to("cuda" if torch.cuda.is_available() else "cpu"), y_batch.to("cuda" if torch.cuda.is_available() else "cpu")
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()

    # 학습된 모델 저장
    models[target_col] = model

    # 미래 예측
    future_predictions = []
    last_sequence = X_test[-1].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            pred = model(last_sequence)
            next_value = pred.squeeze().item()
            next_value = max(0, next_value)  # 음수 방지
            future_predictions.append(next_value)

            # 새로운 입력 데이터 구성
            pred_expanded = torch.tensor([[next_value] * df.shape[1]], dtype=torch.float32).unsqueeze(0).to(last_sequence.device)
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred_expanded), dim=1)

    # 결과 저장
    future_results[target_col] = future_predictions

# Step 7. 모든 Date (과거 + 미래)에 대한 예측 시각화
plt.figure(figsize=(20, 10))

# 색상 매핑
colors = {
    "Close BTC-USD": "blue",
    "Close MTL-USD": "orange",
    "Close USDT-USD": "green",
    "Close GT-USD": "red",
    "Close FLUX-USD": "purple"
}

for target_col in target_columns:
    print(f"Predicting for all dates (past + future) for {target_col}...")

    # 과거 데이터 (모델 예측 수행)
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(df) - look_back):  # 기존 데이터에 대한 예측
            input_seq = torch.tensor(df.values[i:i + look_back], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            pred = model(input_seq)
            all_predictions.append(pred.squeeze().item())

        # 미래 데이터 예측
        last_sequence = torch.tensor(df.values[-look_back:], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        for _ in range(future_steps):
            pred = model(last_sequence)
            next_value = pred.squeeze().item()
            all_predictions.append(next_value)

            # 새로운 입력 데이터 구성
            pred_expanded = torch.tensor([[next_value] * df.shape[1]], dtype=torch.float32).unsqueeze(0).to(last_sequence.device)
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred_expanded), dim=1)

    # 전체 Date 생성
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
    all_dates = df.index.tolist() + future_dates.tolist()

    # 시각화 데이터프레임 생성
    prediction_df = pd.DataFrame({'Date': all_dates, 'Predicted': all_predictions}).set_index('Date')

    # 그래프 그리기
    color = colors.get(target_col, "black")
    plt.plot(df.index, df[target_col], label=f"Actual {target_col}", color=color, alpha=0.6)  # 실제 값
    plt.plot(prediction_df.index, prediction_df['Predicted'], linestyle='dotted', label=f"Predicted {target_col}", color=color, alpha=0.8)  # 예측 값

# 그래프 설정
plt.legend()
plt.title('LSTM Prediction for All Dates (Past + Future)')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.grid(True)
plt.show()
