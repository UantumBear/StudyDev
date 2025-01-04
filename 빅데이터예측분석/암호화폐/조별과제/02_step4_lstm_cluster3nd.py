import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import hyperparameter as hp
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from sklearn.metrics import r2_score


# Step GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Step 한글 폰트 경로 지정
font_path = 'fonts/Paperlogy-4Regular.ttf'
font_prop = fm.FontProperties(fname=font_path, size=8) # 폰트 프로퍼티 설정
# matplotlib의 rcParams 설정을 통해 전역적으로 한글 폰트 적용
plt.rcParams['font.family'] = font_prop.get_name()

# Step 모델 셋팅
setting = f"epoch={hp.EPOCHS}, batch={hp.BATCH_SIZE}, lr={hp.LEARNING_RATE}, hidden_size={hp.HIDDEN_SIZE}, num_layer={hp.NUM_LAYERS}, drop_out={hp.DROP_OUT}, patience={hp.PATIENCE}, look_back={hp.LOOK_BACK}"

# Step 데이터 불러오기 (원본 데이터)
file_path = 'data/output/02_cluster_2_for_train.xlsx'
df = pd.read_excel(file_path)

# 2. 날짜 열 제외 및 정렬
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')  # 날짜 순 정렬
dates = df['Date']  # 날짜 저장

# Step 타겟 데이터 설정
# 특성 확인
target_column = "BSV-USD_Close"
df[target_column].hist(bins=50)
plt.title("Original Distribution")
# plt.show()
# 실제 날짜
actual_dates = df['Date']
actual_values = df[target_column]
# 예측하고자 하는 날짜
last_date = actual_dates.iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]  # 365일 생성


# Step momalize. 각 열별 로그변환 후 데이터 정규화
# 각 열별 로그 변환 수행
for column in df.columns:
    if column != "Date":  # 날짜 열은 제외
        df[column] = np.log1p(df[column])  # log(1 + x) 변환

# 로그 변환된 데이터 확인
print("로그 변환된 target 컬럼 출력: \n", df[target_column])

# MinMaxScaler 객체를 만들 때 feature_range 확인/설정
scaler = MinMaxScaler(feature_range=(0, 1))  # 기본값
print(f"정규화 범위: scaler.feature_range: {scaler.feature_range}")  # (0, 1) 출력

scalers = {}
for column in df.columns:
    if column != "Date":
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[column] = scaler.fit_transform(df[[column]])
        scalers[column] = scaler  # 스케일러 저장 (역정규화 시 사용)

# 데이터 확인용
print(df[target_column].describe())
df[target_column].hist(bins=50)
plt.title("Log Transformed Distribution")
# plt.show()

# 로그 변환 후 정규화된 target 컬럼 출력
print("로그 변환 후 정규화된 target 컬럼 출력: \n", df[target_column])
# df[[target_column]].to_csv("data/output/로그변환후정규화한BSV.csv", index=False)
df[[target_column]].to_excel("data/output/로그변환후정규화한BSV.xlsx", index=False)
# data = df.drop(columns=['Date']).copy()  # 정규화 대상 데이터


# Fx. 역정규화와 역로그변환 함수
def inverse_transform(predictions, scaler_target, log_transformed=True):
    # 정규화 역변환
    predictions_inverse = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
    # 로그 변환을 적용했을 경우, 역 로그 변환
    if log_transformed:
        predictions_inverse = np.expm1(predictions_inverse)
    return predictions_inverse

# 다시 값을 되돌린 target 컬럼 출력:
test_result = inverse_transform(df[target_column].values, scalers[target_column])
print("로그 변환 후 정규화된 target 컬럼 출력: \n", df[target_column])








# Step 3. 학습/검증/테스트 데이터 분할 준비
# 로그 변환 + 정규화된 데이터를 numpy 배열로 변환
data_scaled = df.drop(columns=['Date']).values  # 'Date' 열 제외 후 numpy 배열로 변환
train_size = int(len(data_scaled) * 0.7)
val_size = int(len(data_scaled) * 0.15)
train_data = data_scaled[:train_size]
val_data = data_scaled[train_size:train_size + val_size]
test_data = data_scaled[train_size + val_size:]




target_column_index = df.columns.get_loc("BSV-USD_Close")
look_back = hp.LOOK_BACK

# 4. 시계열 데이터 준비
def create_sequences(data, look_back, target_column_index):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, :])  # 전체 피처 사용
        y.append(data[i + look_back, target_column_index])  # 특정 타겟 열만 사용
    return np.array(X), np.array(y)


# 시계열 데이터 생성
X_train, y_train = create_sequences(train_data, look_back, target_column_index)
X_val, y_val = create_sequences(val_data, look_back, target_column_index)
X_test, y_test = create_sequences(test_data, look_back, target_column_index)

# 데이터 확인용
print("First 5 sequences of y_train:")
print(y_train[:5])  # y_train 값 확인
print("First 10 values of y_test:")
print(y_test[:10])  # y_test 값 확인

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
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=hp.DROP_OUT)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM 출력
        out = self.dropout(out[:, -1, :])  # Dropout 적용 (마지막 타임스텝의 출력)
        out = self.fc(out)  # Fully Connected 레이어

        return out


input_size = X_train.shape[2]
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
output_size = hp.OUTPUT_SIZE

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 매 10 에포크마다 학습률을 절반으로 감소

log_file_path = rf"data/model/train_log_cluster3nd_{hp.VER}.txt"
# 7. 모델 학습
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    # # 로그 파일 초기화
    # if os.path.exists(log_file_path):
    #     os.remove(log_file_path)  # 기존 파일 삭제
    # with open(log_file_path, "w") as log_file:
    #     log_file.write("Epoch,Train Loss,Val Loss,Train MSE,Train MAE,Val MSE,Val MAE\n")  # 헤더 작성
    # 로그 파일 초기화
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch,Train Loss,Val Loss,Train MSE,Train MAE,Val MSE,Val MAE,Val R2,Val MAPE\n")  # 헤더 작


    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # 초기 Best Validation Loss 값
    patience_counter = 0  # Patience Counter 초기화

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], [] # 성능 평가를 위함


        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 성능 평가를 위함 - 예측값과 실제값 저장
            train_preds.extend(outputs.squeeze().cpu().detach().numpy())
            train_targets.extend(y_batch.cpu().numpy())
            #


        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], [] # 성능 평가를 위함

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

                # 예측값과 실제값 저장
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        scheduler.step()  # 학습률 업데이트

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        # MSE, MAE 계산
        # train_mse = mean_squared_error(train_targets, train_preds)
        # train_mae = mean_absolute_error(train_targets, train_preds)
        # val_mse = mean_squared_error(val_targets, val_preds)
        # val_mae = mean_absolute_error(val_targets, val_preds)

        # RMSE 계산 (정규화된 값으로 계산)
        # train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        # val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        # 로그변환 및 정규화 원복
        # train_targets_inverse = inverse_transform(np.array(train_targets), scalers[target_column])
        # train_preds_inverse = inverse_transform(np.array(train_preds), scalers[target_column])
        # val_targets_inverse = inverse_transform(np.array(val_targets), scalers[target_column])
        # val_preds_inverse = inverse_transform(np.array(val_preds), scalers[target_column])
        # 원복된 스케일 값으로 MSE, RMSE 계산
        # train_mse_inverse = mean_squared_error(train_targets_inverse, train_preds_inverse)
        # val_mse_inverse = mean_squared_error(val_targets_inverse, val_preds_inverse)
        # train_rmse_inverse = np.sqrt(mean_squared_error(train_targets_inverse, train_preds_inverse))
        # val_rmse_inverse = np.sqrt(mean_squared_error(val_targets_inverse, val_preds_inverse))

        # R², MAPE 추가 계산
        # val_r2 = r2_score(val_targets, val_preds)
        # val_mape = np.mean(np.abs((np.array(val_targets) - np.array(val_preds)) / np.array(val_targets))) * 100

        # Epoch 로그 출력
        epoch_log = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f} | "
            # f"Train MSE(로그변환,정규화스케일): {train_mse:.4f}, Train MAE: {train_mae:.4f} | "
            # f"Train RMSE(로그변환,정규화스케일): {train_rmse:.4f}, Train MAE: {val_rmse:.4f} | "
            # f"Train MSE(원본스케일): {train_mse_inverse:.4f}, Train MAE: {val_mse_inverse:.4f} | "
            # f"Train RMSE(원본스케일): {train_rmse_inverse:.4f}, Train MAE: {val_rmse_inverse:.4f} | "
            # f"Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f} | "
            # f"Val R²: {val_r2:.4f}, Val MAPE: {val_mape:.2f}%"
        )
        print(epoch_log)

        # 로그 파일 저장
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch + 1},{train_losses[-1]:.4f},{val_losses[-1]:.4f}\n"
            )

        # print(
        #     f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        # )

        # Early Stopping 로직
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0  # 성능 개선 시 patience 카운터 초기화
            model_path = rf'data/model/best_model_cluster3nd_{hp.VER}.pth'
            torch.save(model.state_dict(), model_path)  # 모델 상태 저장
            print(f"Epoch {epoch + 1}: Validation loss improved to {best_val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch + 1}: No improvement in validation loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best Validation Loss: {best_val_loss:.4f}")
                break

    # 학습 및 검증 손실 시각화 (학습 종료 후 실행)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"학습검증손실:: {setting}")
    # 그래프 저장
    plt.savefig(rf'data/model/02_step4_lstm_손실_cluster3nd_{hp.VER}.png', dpi=300, bbox_inches='tight')  # 파일로 저장
    plt.close()  # 플롯 닫기 (메모리 절약)

# train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=hp.EPOCHS, patience=hp.PATIENCE)

# 모델 불러오기
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(rf'data/model/best_model_cluster3nd_{hp.VER}.pth'))
model.eval()  # 평가 모드로 전환
print("Model loaded and ready for prediction.")


# Step 1: 2019년~2024년 학습 데이터 기반 예측
data_scaled = df.drop(columns=['Date']).values  # 정규화된 데이터
X_full, _ = create_sequences(data_scaled, look_back, target_column_index)  # 전체 시퀀스 생성
X_full = torch.tensor(X_full, dtype=torch.float32).to(device)

# 과거 데이터 예측
with torch.no_grad():
    past_predictions = model(X_full).cpu().numpy().flatten()
past_predictions_inverse = inverse_transform(past_predictions, scalers[target_column])  # 복원

# 과거 예측 데이터에 맞는 날짜
adjusted_past_dates = df['Date'][look_back:]  # look_back 이후의 날짜만 사용

# Step 2: 2025년 이후 미래 데이터 예측
future_input = data_scaled[-look_back:].copy()  # 마지막 윈도우 데이터
future_predictions = []

for _ in range(365):  # 미래 365일 예측
    future_input_tensor = torch.tensor(future_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        future_pred = model(future_input_tensor).cpu().numpy().flatten()
    future_predictions.append(future_pred[0])

    # 슬라이딩 윈도우 업데이트
    future_input = np.append(future_input[1:], [[future_pred[0]] + [0] * (future_input.shape[1] - 1)], axis=0)

future_predictions_inverse = inverse_transform(np.array(future_predictions), scalers[target_column])  # 복원

# Step 3: 날짜 생성
past_dates = adjusted_past_dates  # 과거 데이터는 adjusted_past_dates 사용
future_dates = [past_dates.iloc[-1] + pd.Timedelta(days=i) for i in range(1, 366)]  # 미래 날짜
all_dates = pd.concat([past_dates, pd.Series(future_dates)], ignore_index=True)

# Step 4: 과거와 미래 예측 결합
all_predictions = np.append(past_predictions_inverse, future_predictions_inverse)

# Step 5: 그래프 그리기
plt.figure(figsize=(15, 7))

# 학습 데이터 기반 예측
plt.plot(past_dates, past_predictions_inverse, label="예측 값 (과거-현재)", linestyle='--', color="orange")
# 원본 데이터
plt.plot(df['Date'], actual_values, label="실제 값", color="blue")

# 미래 데이터 예측
plt.plot(future_dates, future_predictions_inverse, label="예측 값 (현재+)", linestyle='--', color="green")

plt.legend()
plt.title("Predictions for 2019 to 2025")
plt.xlabel("Date")
plt.ylabel(target_column)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(rf'data/model/02_BSV_예측_2019_2025_cluster3nd_{hp.VER}.png', dpi=300)
plt.show()

############################### 성능검증





def evaluate_model(model, test_loader, scaler, target_column, log_transformed=True):
    model.eval()
    test_preds, test_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_preds.extend(outputs.squeeze().cpu().numpy())
            test_targets.extend(y_batch.cpu().numpy())

    # 2. 정규화 상태에서 MSE, RMSE 계산
    test_mse_normalized = mean_squared_error(test_targets, test_preds)
    test_rmse_normalized = np.sqrt(test_mse_normalized)
    test_mae_normalized = mean_absolute_error(test_targets, test_preds)

    # 3. 역정규화 및 역로그변환
    test_preds_inverse = inverse_transform(np.array(test_preds), scaler, log_transformed)
    test_targets_inverse = inverse_transform(np.array(test_targets), scaler, log_transformed)

    # 4. 원본 스케일에서 MSE, RMSE 계산
    test_mse_original = mean_squared_error(test_targets_inverse, test_preds_inverse)
    test_rmse_original = np.sqrt(test_mse_original)
    test_mae_original = mean_absolute_error(test_targets_inverse, test_preds_inverse)
    test_r2_original = r2_score(test_targets_inverse, test_preds_inverse)
    test_mape_original = np.mean(np.abs((test_targets_inverse - test_preds_inverse) / test_targets_inverse)) * 100

    # 5. 결과 출력
    print("Test Results (Normalized):")
    print(f" - MSE: {test_mse_normalized:.4f}")
    print(f" - RMSE: {test_rmse_normalized:.4f}")
    print(f" - MAE: {test_mae_normalized:.4f}")
    print("\nTest Results (Original Scale):")
    print(f" - MSE: {test_mse_original:.4f}")
    print(f" - RMSE: {test_rmse_original:.4f}")
    print(f" - MAE: {test_mae_original:.4f}")
    print(f" - R²: {test_r2_original:.4f}")
    print(f" - MAPE: {test_mape_original:.2f}%")

    # 6. 결과 저장
    with open(rf'data/model/test_metrics_{hp.VER}.txt', 'w') as log_file:
        log_file.write("Test Results (Normalized):\n")
        log_file.write(f"MSE: {test_mse_normalized:.4f}\n")
        log_file.write(f"RMSE: {test_rmse_normalized:.4f}\n")
        log_file.write(f"MAE: {test_mae_normalized:.4f}\n")
        log_file.write("\nTest Results (Original Scale):\n")
        log_file.write(f"MSE: {test_mse_original:.4f}\n")
        log_file.write(f"RMSE: {test_rmse_original:.4f}\n")
        log_file.write(f"MAE: {test_mae_original:.4f}\n")
        log_file.write(f"R²: {test_r2_original:.4f}\n")
        log_file.write(f"MAPE: {test_mape_original:.2f}%\n")

    return {
        "Normalized": {"MSE": test_mse_normalized, "RMSE": test_rmse_normalized, "MAE": test_mae_normalized},
        "Original": {
            "MSE": test_mse_original,
            "RMSE": test_rmse_original,
            "MAE": test_mae_original,
            "R2": test_r2_original,
            "MAPE": test_mape_original,
        },
    }
model.eval()

# 테스트 데이터 평가
# 테스트 데이터 평가
results = evaluate_model(model, test_loader, scalers[target_column], target_column)

# 최종 결과 출력
print("\nComparison of Test Results:")
print("Normalized Results:")
print(f" - MSE: {results['Normalized']['MSE']:.4f}")
print(f" - RMSE: {results['Normalized']['RMSE']:.4f}")
print(f" - MAE: {results['Normalized']['MAE']:.4f}")
print("Original Scale Results:")
print(f" - MSE: {results['Original']['MSE']:.4f}")
print(f" - RMSE: {results['Original']['RMSE']:.4f}")
print(f" - MAE: {results['Original']['MAE']:.4f}")
print(f" - R²: {results['Original']['R2']:.4f}")
print(f" - MAPE: {results['Original']['MAPE']:.2f}%")
