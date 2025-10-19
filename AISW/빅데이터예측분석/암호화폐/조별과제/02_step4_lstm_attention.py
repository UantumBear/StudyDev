import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import hyperparameter as hp

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 데이터 불러오기
file_path = 'data/output/02_cluster_0_normalized_train_data.xlsx'
df = pd.read_excel(file_path)

# 2. 날짜 열 제외 및 정렬
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')  # 날짜 순 정렬
data = df.drop(columns=['Date']).copy()  # 모든 열 사용

# 3. 학습/검증/테스트 데이터 분할
data = data.values  # Numpy 배열로 변환
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]


# 4. 시계열 데이터 준비
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, :])
        y.append(data[i + look_back, 0])  # LUNC-USD_Close 예측
    return np.array(X), np.array(y)


look_back = 200
X_train, y_train = create_sequences(train_data, look_back)
X_val, y_val = create_sequences(val_data, look_back)
X_test, y_test = create_sequences(test_data, look_back)

# PyTorch Tensor로 변환
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# DataLoader 생성
batch_size = hp.BATCH_SIZE
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# 5. GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 6. LSTM 모델 정의
# LSTM 모델에 Attention 추가
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)  # Attention 스코어 계산
        self.fc = nn.Linear(hidden_size, output_size)  # Fully Connected Layer
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        # LSTM 출력
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]

        # Attention 스코어 계산
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch_size, seq_len, 1]

        # Attention 가중합 계산
        attn_applied = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_size]

        # Dropout 적용
        attn_applied = self.dropout(attn_applied)  # [batch_size, hidden_size]

        # Fully Connected Layer로 최종 출력
        out = self.fc(attn_applied)  # [batch_size, output_size]
        return out

    def init_weights(self):
        # 가중치 초기화 (Xavier 초기화)
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'attention' in name or 'fc' in name:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.0)


input_size = X_train.shape[2] # 데이터의 feature 개수 (X_train의 마지막 차원)
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
output_size = hp.OUTPUT_SIZE

model = LSTMWithAttention(input_size, hidden_size, num_layers, output_size).to(device)
model.init_weights()  # 가중치 초기화
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)


# 7. 모델 학습
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")


train_model(model, train_loader, val_loader, criterion, optimizer, epochs=hp.EPOCHS)

# 8. 예측 및 결과 시각화
# model.eval()
# predictions, actual = [], []
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch = X_batch.to(device)
#         outputs = model(X_batch)
#         predictions.append(outputs.squeeze().cpu().numpy())
#         actual.append(y_batch.numpy())
#
# predictions = np.concatenate(predictions)
# actual = np.concatenate(actual)
#
# plt.figure(figsize=(14, 5))
# plt.plot(actual, label="Actual Prices")
# plt.plot(predictions, label="Predicted Prices")
# plt.legend()
# plt.show()

# 8. 예측 및 결과 시각화
test_dates = df['Date'][train_size+val_size+look_back:]  # 테스트 데이터에 해당하는 날짜 추출

model.eval()
predictions, actual = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.append(outputs.squeeze().cpu().numpy())
        actual.append(y_batch.numpy())

predictions = np.concatenate(predictions)
actual = np.concatenate(actual)

plt.figure(figsize=(14, 5))
plt.plot(test_dates, actual, label="Actual Prices", color="blue")
plt.plot(test_dates, predictions, label="Predicted Prices", color="orange")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)  # 날짜가 겹치지 않도록 회전
plt.show()

# Attention 스코어 시각화
with torch.no_grad():
    model.eval()  # 평가 모드 전환
    for X_batch, _ in train_loader:  # 학습 데이터의 첫 번째 배치 사용
        X_batch = X_batch.to(device)

        # LSTM 출력
        lstm_out, _ = model.lstm(X_batch)

        # Attention 스코어 계산
        attn_weights = F.softmax(model.attention(lstm_out), dim=1)  # Attention 스코어

        # Attention 스코어 시각화
        plt.imshow(attn_weights[0].cpu().numpy(), cmap='hot', aspect='auto')
        plt.colorbar()
        plt.title("Attention Weights")
        plt.xlabel("Sequence Step")
        plt.ylabel("Attention Weight")
        plt.show()
        break  # 한 번만 시각화