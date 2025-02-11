import pandas as pd
from sklearn.model_selection import train_test_split # train, test 셋 나누기
from sklearn.linear_model import LinearRegression # 선형회귀
import statsmodels.api as sm # 회귀분석 요약리포트를 위함
from sklearn.metrics import mean_squared_error # MSE
from sklearn.tree import DecisionTreeRegressor # 결정 트리를 위함
from sklearn import tree  # 트리 구조 시각화를 위한 모듈
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import MinMaxScaler # 정규화를 위함
from sklearn.neural_network import MLPRegressor  # MLP( Multi-Layer Perceptron),  인공 신경망 모델 대체를 위함
import tensorflow as tf # 다층신경망 nerualnet 대체를 위함
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor # 랜덤 포레스트 모델 사용을 위함
import numpy as np


from colorama import Fore, Style

def red(redtext, data=None):
    print("\n" + Fore.RED + redtext + Style.RESET_ALL )
    if data is not None:
        print(data)
def green(greentext, data=None):
    print("\n" + Fore.GREEN + greentext + Style.RESET_ALL)
    if data is not None:
        print(data)



# Step 1. Boston Dataset 로드
"""R
library(MASS) # MASS 패키지 로드
data(Boston) # Boston dataset 로드
"""
# CSV 파일 불러오기, R에서 사용한 데이터를 추출하였음.
boston_df = pd.read_csv('boston_data.csv')



# Step 2. Boston Dataset 을 train/test set 으로 분할
"""R, 69p
idx <- sample(1:nrow(Boston), size=nrow(Boston)*0.7, replace=F)
Boston_train <- Boston[idx, ] # 훈련용 데이터 세트 분할 (idx 를 사용해 Boston 데이터에서 해당 인덱스의 값들을 추출)
Boston_test <- Boston[-idx, ] # 평가용 데이터 세트 분할 (idx 포함되지 않은 데이터들을 추출)
dim(Boston_train) ; dim(Boston_test)
"""
# 훈련 데이터와 테스트 데이터 분할 (70% 훈련, 30% 테스트)
# train_test_split 이 바로 랜덤 분할 하는 기능을 제공하기 때문에 굳이 idx 로 랜덤 인덱스를 생성할 필요가 없다.
# 만약 생성하고자 한다면, 아래와 같이 직접 분할하여도 된다.
# idx = np.arange(len(df))  # 인덱스 생성
# X = df.drop('MEDV', axis=1)
# y = df['MEDV']
# X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, test_size=0.3, random_state=1)
###Python, 69p

train_df, test_df = train_test_split(boston_df, test_size=0.3, random_state=1)



# Step 3. Boston Dataset 의 train/test 데이터 확인 및 정리
"""R, 69p
dim(Boston_train) ; dim(Boston_test) # 데이터 프레임의 행과 열의 개수를 반환
>> 결과: [1] 354  14 , [1] 152  14
"""
### Python, 69p 데이터 차원 확인
red("train_df.shape", train_df.shape) # 결과: 354  14
red("test_df.shape", test_df.shape)   # 결과: 152  14

# R에서는 모델을 만들때 lm('medv' ... ) 이런 형태로 종속변수를 나누고 자동으로 분석 하기 때문에 medv 를 제거하지 않고 train,test set 을 나누어도 되지만
# python 에서는 종속변수(목표변수)는 제거한 채 학습 데이터 셋을 나누어야 한다. (당연한 것 같다.. x데이터, y 데이터니까)
X_train = train_df.drop('medv', axis=1)  # 독립 변수
y_train = train_df['medv']  # 종속 변수
X_test = test_df.drop('medv', axis=1)  # 테스트 독립 변수
y_test = test_df['medv']  # 테스트 종속 변수



# Step 4. 선형 회귀 모델 적합 및 분석 결과 출력
"""R, 69p
lm.fit <- lm(medv~., data=Boston_train)
summary(lm.fit)
"""
lm = LinearRegression() # 선형 회귀 모델 적합
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test) # 예측 수행
red("y_pred", y_pred) # 확률 출력

X_train_const = sm.add_constant(X_train) # 종속 변수에 상수항(절편)을 추가
model_basic = sm.OLS(y_train, X_train_const)  # statsmodels로 회귀 모델 적합 # OLS: Ordinary Least Squares
results_basic = model_basic.fit()
red("results_basic.summary()", results_basic.summary()) # 회귀 분석 결과 출력 (R의 summary(lm.fit)와 유사)


# Step 5. 변수 선택 방식을 변경하여 선형 회귀 모델을 적합 및 분석 결과 출력
"""R, 70p
lm.fit2 <- step(lm.fit, method="both") # 변수 선택을 자동으로 수행하여 더 나은 모델을 찾음
summary(lm.fit2)
Fx 1. forward_selection 과 backward_elimination 은 R 코드만 보여주고 같은 소스를 요청해서 얻은 함수
Fx 2. stepwise_selection 은 R 코드와 결과물까지 보여주고 소스를 요청해서 얻은 함수
"""
### Python, 70p Fx 1-1. 전진 선택 방식 - 변수를 하나씩 추가하면서 모델 성능이 향상되는지 확인하는 방법
def forward_selection(X, y, significance_level=0.05):
    # significance_level: 변수 선택 기준이 되는 유의 수준 (기본값 0.05)
    initial_features = []
    while True:
        remaining_features = [p for p in X.columns if p not in initial_features]
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[initial_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_feature = new_pval.idxmin()
            initial_features.append(best_feature)
        else:
            break
    return initial_features

# 전진 선택 함수 적용 (X_train, y_train 사용)
forward_selected_features = forward_selection(X_train, y_train)
print(f"선택된 변수들: {forward_selected_features}")

# 선택된 변수로 회귀 모델 적합
X_train_forward_selected = sm.add_constant(X_train[forward_selected_features])  # 상수항 추가
model_forward = sm.OLS(y_train, X_train_forward_selected).fit()
print(f"model_forward.summary(): \n{model_forward.summary()}")
### Python, 70p 후진 제거 함수 - 모든 변수를 포함한 상태에서 시작하여, 통계적으로 유의하지 않은 변수를 하나씩 제거하는 방법
def backward_elimination(X, y, significance_level=0.05):
    features = list(X.columns)
    while len(features) > 0:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        p_values = model.pvalues.iloc[1:]  # 절편을 제외한 p-value 값들
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

# 후진 제거 함수 적용 (X_train, y_train 사용)
backward_selected_features = backward_elimination(X_train, y_train)
print(f"선택된 변수들 (후진 제거): {backward_selected_features}")

# 선택된 변수로 회귀 모델 적합
X_train_backward_selected = sm.add_constant(X_train[backward_selected_features])  # 상수항 추가
backward_model = sm.OLS(y_train, X_train_backward_selected).fit()
print(f"backward_model.summary(): \n{backward_model.summary()}")

# 위 부분이 R 과 결과가 다르다. TODO 이유가 무엇일까?
# 일단 여러 차례 GPT에게 물어보았다.
# 70 p Fx 2-1. 전진 후진 선택 - lm.fit2 을 보여주고 얻은 함수
# def stepwise_selection_basic(X, y, direction='forward', scoring='neg_mean_squared_error'):
#     lr = LinearRegression()
#
#     # 전진 선택 또는 후진 제거 설정
#     if direction == 'forward':
#         forward = True
#     elif direction == 'backward':
#         forward = False
#     else:
#         raise ValueError("direction must be 'forward' or 'backward'")
#
#     # Stepwise feature selection 수행
#     sfs = SFS(lr,
#               k_features="best",
#               forward=forward,
#               floating=False,
#               scoring=scoring,
#               cv=0)
#
#     # Fit the selector to the data
#     sfs.fit(X, y)
#
#     # Get selected features
#     selected_features = list(sfs.k_feature_names_)
#     print("Selected features:", selected_features)
#
#     # Fit the final model with selected features
#     X_selected = X[selected_features]
#     model = sm.OLS(y, sm.add_constant(X_selected)).fit()
#
#     # Return the fitted model and selected features
#     return model, selected_features
#
# model_stepwise_forward, selected_features_stepwise_forward = stepwise_selection_basic(X_train, y_train, direction='forward')
# red("Fx 2-1. 'stepwise forward' selected_features", selected_features_stepwise_forward)
# red("Fx 2-1. 'stepwise forward' model.summary()", model_stepwise_forward.summary()) # 모델 결과 출력
#
# model_stepwise_backward, selected_features_stepwise_backward = stepwise_selection_basic(X_train, y_train, direction='backward')
# red("Fx 2-1. 'stepwise backward' selected_features", selected_features_stepwise_backward)
# red("Fx 2-1. 'stepwise backward' model.summary()", model_stepwise_backward.summary()) # 모델 결과 출력



# Fx 2-2. p-value를 기반으로 양방향(stepwise) 선택을 수행, (summary(lm.fit2) 의 결과까지 보여주고 얻은 식)
# def stepwise_selection_p(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.1):
#     # threshold_in: 변수 추가 기준,  threshold_out: 변수 제거 기준
#     included = list(initial_list)
#     while True:
#         changed = False
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(index=excluded, dtype=float)
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()
#
#         if worst_pval > threshold_out:
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             changed = True
#
#         print(f"Included features: {included}")  # 중간 로그 추가
#         print(f"Best p-value: {best_pval}, Worst p-value: {worst_pval}")
#
#         if not changed:
#             break
#
#     return included
#
#
# resulting_features_p = stepwise_selection_p(X_train, y_train) # 양방향 선택 수행
# model_p = sm.OLS(y_train, sm.add_constant(X_train[resulting_features_p])).fit() # 선택된 변수들로 최종 모델 적합
#
# red("Fx 2-2. [p-value 기반 양방향] selected_features_aic", resulting_features_p)
# red("Fx 2-2. [p-value 기반 양방향] model_aic.summary()", model_p.summary()) # 모델 결과 출력


"""
비슷한 결과가 나오지 않는다. 어떤 차이가 있는 걸까?
R 
 lm.fit2 <- step(lm.fit, method="both") :: both: 양방향
 summary(lm.fit2)
요소 1) AIC
 R 의 step() 함수는 기본적으로 AIC(Akaike Information Criterion) 을 기준으로 변수를 선택한다고 한다.
 Python 에서는 statsmodels 을 통해 AIC 를 계산할 수 있지만, mlxtend의 SequentialFeatureSelector 는 
 기본적으로 기본적으로 scoring metric 를 사용하지 않고 neg_mean_squared_error 를 사용한다고 한다. 
요소 2) 데이터 처리 방식
 R에서는 결측치와 범주형 변수를 자동으로 처리해주지만, Python은 직접 데이터 전처리를 해야할 수 있다.

"""

# Fx 2-3. AIC 기반 양방향
def stepwise_selection_aic(X, y, initial_list=[]):

    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_aic = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_aic[new_column] = model.aic
        best_aic = new_aic.min()

        # 이전 모델의 AIC와 비교하여 작은 경우에만 추가
        current_aic = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit().aic
        if best_aic < current_aic:
            best_feature = new_aic.idxmin()
            included.append(best_feature)
            changed = True

        # backward step (AIC 기반으로 변수 제거)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        aic_with_feature_removed = pd.Series(index=included, dtype=float)
        for feature in included:
            included_minus_feature = [f for f in included if f != feature]
            model_removed = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included_minus_feature]))).fit()
            aic_with_feature_removed[feature] = model_removed.aic
        worst_aic = aic_with_feature_removed.min()

        # AIC가 감소하지 않는다면 해당 변수 제거
        if worst_aic < current_aic:
            worst_feature = aic_with_feature_removed.idxmin()
            included.remove(worst_feature)
            changed = True

        if not changed:
            break

    return included  # 선택된 변수만 반환




# AIC 기준 양방향 선택
resulting_features_aic = stepwise_selection_aic(X_train, y_train)
model_aic = sm.OLS(y_train, sm.add_constant(X_train[resulting_features_aic])).fit()

red("Fx 2-3. [AIC 기반 양방향] selected_features_aic", resulting_features_aic)
red("Fx 2-3. [AIC 기반 양방향] model_aic.summary()", model_aic.summary()) # 모델 결과 출력

## 결과가 같지는 않지만 해당 모델이 R 과 마찬가지로 AIC 기반 양방향 변수 선택 방법 이므로, 아래 소스 부터는 R에서의 lm 을 해당 모델로 사용해야겠다.




# Step 6. 테스트 데이터에 대한 Predict 및 MSE(평균제곱오차)를 사용한 모델 성능 평가
# 전진 선택, 후진 제거 방식으로 선택된 변수를 사용하여 test data 에 대해 예측하고,
# 예측값과 실제값의 평균제곱오차(MSE)를 계산하여 모델 성능을 평가함.

"""R, 71p
lm.yhat2 <- predict(lm.fit2, newdata=Boston_test) # Boston_test 데이터를 입력으로 받아, 예측값 생성
mean((lm.yhat2-Boston_test$medv)^2) # 예측값과 실제값간의 평균제곱오차(MSE) 계산
"""
# 위에서 만들었었던, 포워드, 백워드, aic 모델 세개 비교 용도..
# 테스트 데이터에 대한 예측 (전진 선택 방식으로 선택된 변수 사용)
X_test_forward_selected = sm.add_constant(X_test[forward_selected_features])  # 상수항 추가
y_pred_forward = model_forward.predict(X_test_forward_selected)  # 예측값 생성

# 평균제곱오차(MSE) 계산
mse_forward = mean_squared_error(y_test, y_pred_forward)
red("Forward Selection Model MSE:", mse_forward)

# 테스트 데이터에 대한 예측 (후진 제거 방식으로 선택된 변수 사용)
X_test_backward_selected = sm.add_constant(X_test[backward_selected_features])  # 상수항 추가
y_pred_backward = backward_model.predict(X_test_backward_selected)  # 예측값 생성

# 평균제곱오차(MSE) 계산
mse_backward = mean_squared_error(y_test, y_pred_backward)
red("Backward Elimination Model MSE: ", mse_backward)

# AIC 기반으로 선택된 변수 사용
X_test_aic_selected = sm.add_constant(X_test[resulting_features_aic])  # AIC로 선택된 변수 추가
y_pred_aic = model_aic.predict(X_test_aic_selected)  # AIC 기반 모델로 예측 수행

# 평균제곱오차(MSE) 계산
mse_aic = mean_squared_error(y_test, y_pred_aic)
red("AIC-based Model MSE: ", mse_aic)

## R 에서 MSE 는 17.88976 로, aic 기반 모델의 MSE가 가장 비슷하다.




# 이제까지 회귀 분석 모델이었고, 이젠
# Step 7. 결정 트리 모델 적합, 요약 분석, 시각화
"""R,  72p
# 기본 tree 모델
library(tree)
tree.fit <- tree(medv~., data=Boston_train)
summary(tree.fit)

plot(tree.fit)
text(tree.fit, pretty=0) # pretty=0: 기본값을 사용하겠다.
"""

# 결정 트리 모델 생성 (기본 설정)
tree_model = DecisionTreeRegressor(max_depth=4, random_state=1)  # R에서 자동 가지치기된 결과가 depth 4여서 4를 적용.
tree_model.fit(X_train, y_train)  # 훈련 데이터로 모델 적합

plt.figure(figsize=(20,10)) # 트리 시각화
tree.plot_tree(tree_model, feature_names=X_train.columns, filled=True)
plt.show()

# Step 8. 결정 트리 모델을 사용한, 테스트 데이터에 대한 예측 및 MSE 계산
"""R, 73p
tree.yhat <- predict(tree.fit, newdata=Boston_test) # 평가 데이터 이용, 예측결과생성
mean((tree.yhat-Boston_test$medv)^2) # 예측값과 실제값간 평균제곱오타(MSE) 계산
"""
y_pred_tree = tree_model.predict(X_test)  # 테스트 데이터에 대한 예측
mse_tree = mean_squared_error(y_test, y_pred_tree)  # MSE 계산
red("Tree Model MSE: ", mse_tree)







# Step 9. 또 다른 R 패키지 rpart 결정 트리 모델
"""R, 73p
library(rpart)
rpart.fit <-rpart(medv~., data=Boston_train)
summary(rpart.fit)
"""
# 파이썬에서는 tree, rpart 를 각각 대체하는 클래스가 없이, 둘 다 DecisionTreeRegressor 를 쓴다고 한다.
# 결정 트리 모델 생성 (rpart와 유사하게 가지치기를 적용하거나 최소 샘플 수를 조정)
rpart_model = DecisionTreeRegressor(max_depth=4, min_samples_split=20, random_state=1)  # 어떻게 해야 비슷한 분석 결과를 내는지 모르겠다..
rpart_model.fit(X_train, y_train)  # 훈련 데이터로 모델 적합
# 트리의 깊이와 리프 노드 개수 확인
red("Tree Depth: ", rpart_model.get_depth())
red("Number of Leaves: ", rpart_model.get_n_leaves())

# 특성 중요도 확인 (summary와 유사한 정보 제공)
red("Feature Importances:")
for name, importance in zip(X_train.columns, rpart_model.feature_importances_):
    print(f"{name}: {importance}")

# Step 10. rpart 결정 트리 모델 시각화
"""R, 74p
library(rpart.plot)
rpart.plot(rpart.fit, digits=3, type=0, extra=1, fallen.leaves=F, cex=1)
"""
plt.figure(figsize=(20,10))
tree.plot_tree(rpart_model, feature_names=X_train.columns, filled=True, rounded=True, precision=3) # rpart_model 시각화
plt.show()

# Step 11. rpart 결정 트리 모델을 사용한, test data 예측 및 MSE 계산
"""R, 75p
rpart.yhat <- predict(rpart.fit, newdata=Boston_test) # 평가 데이터 이용, 예측결과생성
mean((rpart.yhat-Boston_test$medv)^2) # 예측 값과 실제 값간 평균제곱오차(MSE) 계산
"""
y_pred_rpart = rpart_model.predict(X_test)  # 테스트 데이터에 대한 예측
mse_rpart = mean_squared_error(y_test, y_pred_rpart)  # MSE 계산
red("rpart-like Model MSE: ", mse_rpart)


# Step 12. 정규화
"""R, 75p 
normalize <- function(x){ return((x-min(x))/(max(x)-min(x))) } # 정규화 함수 작성
Boston_train_norm <- as.data.frame(sapply(Boston_train, normalize))
Boston_test_norm <- as.data.frame(sapply(Boston_test, normalize))
"""
scaler = MinMaxScaler() # MinMaxScaler 객체 생성 (파이썬에서의 정규화를 위한 클래스)
X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns) # train, test data 에 대해 정규화 수행
X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 검토용
# print("Train Data Range:", X_train_norm.min(), X_train_norm.max())
# print("Test Data Range:", X_test_norm.min(), X_test_norm.max())
## 체크해보니, test 에서는 1을 넘는 범위가 있었다.
## 하지만 GPT에 따르면, 테스트 데이터가 정규화된 후 1을 넘는 값이나 0보다 작은 값을 가지는 것은 일반적으로 문제되지 않는다고 한다..

# Step 13. (정규화 후) 인공 신경망1 (nnet) 모델 생성, 학습, 예측, MES 계산
"""R, 75p
library(nnet)
nnet.fit <- nnet(medv~., data=Boston_train_norm, size=5) # 인공 신경망 적합하기
nnet.yhat <- predict(nnet.fit, newdata=Boston_test_norm, type="raw") # 예측결과생성
mean((nnet.yhat-Boston_test_norm$medv)^2) # 평균제곱오차 계산
"""

mlp_nnet = MLPRegressor(hidden_layer_sizes=(5,),  # 은닉층 뉴런 개수 5개
                   activation='logistic',    # R의 기본 활성화 함수는 logistic (시그모이드)
                   solver='lbfgs',           # R의 nnet에서 사용하는 알고리즘과 유사한 solver 사용
                   max_iter=100,             # R과 동일하게 100번 반복
                   random_state=1)


mlp_nnet.fit(X_train_norm, y_train)  # 정규화된 훈련 데이터로 모델 적합
nnet_yhat = mlp_nnet.predict(X_test_norm)  # 정규화 된 test data 에 대한 예측 수행
mse_nnet = mean_squared_error(y_test, nnet_yhat) # 평균제곱오차(MSE) 계산
red("Neural Network Model MSE (nnet): ", mse_nnet)

## R에서의 학습이 iter 100 value 0.843894, final  value 0.843894 , stopped after 100 iterations 로 100회여서
#  처음에는 다른 조건 없이 학습횟수 100을 입력하자, mlp 에서 아래와 같은 경고가 발생했고, (최적화 되지 않았음을 의미)
# ConvergenceWarning: Stochastic Optimizer: Maximum iterations (2000) reached and the optimization hasn't converged yet.
# MSE는 532.4070998191419 로 나왔다.
# 알아보니, R에서는 활성화 함수로 시그모이드 함수를 기본으로 사용한다고 하여 속성을 추가해주었다. -> 결과 MES: 13.365795136938804


## Step 14. 인공신경망2 (neuralnet)
## nnet 과 neuralnet 어떤 차이가 있을까?
## nnet 은 기본 인공신경망으로, 단일 은닉층 신경망을 지원한다.
## 매개변수 학습에서는 backpropagation 알고리즘을 사용한다.
## neuralnet은 다중 은닉층을 사용한 복잡한 신경망으로,
## 선형 활성화 함수, 비선형 활성화 함수(시그모이드, 탄젠트) 등을 지원한다고 한다.

"""R, 76p
library(neuralnet)
neural.fit <- neuralnet(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio
                        +black+lstat, data=Boston_train_norm, hidden=5) # 인공 신경망 적합하기
neural.results <- compute(neural.fit, Boston_test_norm[1:13]) # 예측결과생성
neural.yhat <- neural.results$net.result
mean((neural.yhat-Boston_test_norm$medv)^2) # 평균제곱오차 계산
"""
model_neuralnet = Sequential([
    Dense(5, input_dim=X_train_norm.shape[1], activation='relu'),  # 은닉층 5개, ReLU 활성화 함수
    Dense(1, activation='linear')  # 출력층 (회귀 문제이므로 활성화 함수 없음)
])
model_neuralnet.compile(optimizer='adam', loss='mean_squared_error') # 모델 컴파일 (손실 함수와 최적화 방법 설정)
model_neuralnet.fit(X_train_norm, y_train, epochs=100, verbose=0) # 모델 학습
y_pred_neuralnet = model_neuralnet.predict(X_test_norm) # 예측
mse_neuralnet = mean_squared_error(y_test, y_pred_neuralnet) # MSE 계산
red("Neural Network Model MSE (neuralnet): ", mse_neuralnet)

# MSE 가 무려 115가 나왔다. y 도 정규화를 해야 하는 걸까?
## 해결 방안 - y 값도 정규화
# y 값도 정규화
# y 값을 numpy 배열로 변환한 후 정규화
scaler_y = MinMaxScaler()
y_train_array = y_train.to_numpy().reshape(-1, 1)  # pandas Series를 numpy 배열로 변환하고 reshape
y_test_array = y_test.to_numpy().reshape(-1, 1)    # pandas Series를 numpy 배열로 변환하고 reshape

y_train_norm = scaler_y.fit_transform(y_train_array)
y_test_norm = scaler_y.transform(y_test_array)

# neuralnet 대체 모델
model_neuralnet2 = Sequential([
    Dense(5, input_dim=X_train_norm.shape[1], activation='sigmoid'),  # 비선형 활성화 함수
    Dense(1, activation='linear')  # 출력층 (회귀 문제이므로 활성화 함수 없음)
])
model_neuralnet2.compile(optimizer='adam', loss='mean_squared_error')  # adam 옵티마이저 사용
model_neuralnet2.fit(X_train_norm, y_train_norm, epochs=100, verbose=0)  # 정규화된 타겟 값으로 학습

# 예측 후 다시 원래 스케일로 복원
y_pred_neuralnet = model_neuralnet2.predict(X_test_norm)
y_pred_neuralnet_rescaled = scaler_y.inverse_transform(y_pred_neuralnet)

# 원래 스케일의 타겟 값과 비교하여 MSE 계산
mse_neuralnet2 = mean_squared_error(y_test_array, y_pred_neuralnet_rescaled)
red("Neural Network Model MSE (neuralnet2):", mse_neuralnet2) ## 최종적으로 tanh: 26 까지는 만들었다..


# 이제까지 일반 회귀 분석 모델, 결정트리, 인공신경망을 사용해 봤다면 이제는 랜덤 포레스트
# Step 15. 랜덤 포레스트 모델 생성, 학습, 정보 출력
"""R, 77p - 78p
library(randomForest)
set.seed(1)
rf.fit <- randomForest(medv~., data=Boston_train, mtry=6, importance=T)
rf.fit

importance(rf.fit)
varImpPlot(rf.fit)
"""
rf_model = RandomForestRegressor(n_estimators=100,  # 기본 100개의 트리 사용
                                 max_features=6,    # R 코드에서 mtry=6이므로 max_features=6
                                 random_state=1)    # R의 set.seed(1)을 반영
rf_model.fit(X_train, y_train) #  학습

importances = rf_model.feature_importances_ # 변수 확인
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6)) # 시각화
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Step 16. 랜덤 포레스트 모델을 이용한 예측 및 MSE 계산
"""R, 79p
rf.yhat <- predict(rf.fit, newdata=Boston_test) # 평가 데이터 이용, 예측결과생성
mean((rf.yhat-Boston_test$medv)^2) # 예측값과 실제값간 평균제곱오차(MSE) 계산
"""
rf_yhat = rf_model.predict(X_test) # 예측
mse_rf = mean_squared_error(y_test, rf_yhat) # MSE
red("Random Forest Model MSE: ", mse_rf)




# TODO
""" R,
## 4장 자율학습 모델

# 91p
iris2 <- iris[, 1:4] # 목표변수(Species) 제외
km.out.withness <- c()
km.out.between <- c()
for (i in 2:7) { # 군집수를 k=2~7 까지 변화시켜가며 클러스터링 시행
   set.seed(1)
  km.out <- kmeans(iris2, centers=i)
   km.out.withness[i-1] <- km.out$tot.withinss # 군집 내 제곱합 저장
   km.out.between[i-1] <- km.out$betweenss # 군집 간 제곱합 저장
}
data.frame(km.out.withness, km.out.between)

# 93p

km.out.k3 <- kmeans(iris2, centers=3)
km.out.k3$centers # 각 군집의 중심점 출력
km.out.k3$cluster # 각 관측치의 할당된 군집번호 출력
km.out.k3$size # 각 군집의 데이터 관측치 개수 출력
table(km.out.k3$cluster, iris$Species) # 군집결과와 원래 품종 개수 비교

plot(
  iris2[, 1:2], 
  col=km.out.k3$cluster, 
  pch=ifelse(km.out.k3$cluster==1, 16, ifelse(km.out.k3$cluster==2, 17, 18)),
  cex=2) ; points(km.out.k3$centers, col=1:3, pch=16:18, cex=5)

# 102p
pc1 <- princomp(USArrests, cor=T) # princomp 함수로 주성분 분석 실시
summary(pc1)

# 103p
pc1$center
pc1$scale
pc1$loadings
pc1$scores
plot(pc1$scores[,1], pc1$scores[,2], xlab="Z1", ylab="Z2")
abline(v=0, h=0, col="gray")

# 105p
biplot(pc1, cex=0.7)
abline(v=0, h=0, col="gray")

# 114p
library(arules)
library(arulesViz)
data(Groceries)

# 115p
data(package = "arules")
Groceries
inspect(Groceries[1:10])

# 116p
summary(Groceries) # Summary 함수로 Groceries 데이터 특성 파악
sort(itemFrequency(Groceries, type="absolute"), decreasing=T)

# 117p
round(sort(itemFrequency(Groceries, type="relative"), decreasing = T), 3)

itemFrequencyPlot(Groceries, topN=10, type="absolut")

# 118p
apriori(Groceries)
result_rules <- apriori(Groceries, parameter=list(support=0.005, confidence=0.5, minlen=2))

# 119p
summary(result_rules)
inspect(result_rules)

# 120p
rules_lift <- sort(result_rules, by="lift", decreasing = T)
inspect(rules_lift[1:5])
rules_conf <- sort(result_rules, by="confidence", decreasing = T)
inspect(rules_conf[1:5])

# 121p
milk_rule <- subset(rules_lift, items %in% "whole milk")
milk_rule
inspect(milk_rule[1:5])

rhs.milk_rule <- subset(rules_lift, rhs %in% "whole milk")
rhs.milk_rule 
inspect(milk_rule[1:5])
wholemilk_rule <- apriori(Groceries, 
                          parameter=list(support=0.005, confidence=0.5, minlen=2), 
                          appearance=list(default="lhs", rhs="whole milk"))
wholemilk_rule <- sort(wholemilk_rule, by="lift", decreasing=T)
inspect(wholemilk_rule[1:5])

# 122p
library(arulesViz)
plot(wholemilk_rule[1:10], method="graph", measure="lift", shading="confidence")


# 135p
library(caret)
idx <- createDataPartition(iris$Species, p=0.7, list=F)
iris_train <- iris[idx, ] # 생성된 인덱스를 이용, 70%의 비율로 학습용 데이터 세트 추출
iris_test <- iris[-idx, ] # 생성된 인덱스 이용, 30%의 비율로 평가용 데이터 세트 추출
table(iris_train$Species)
table(iris_test$Species)

# 136p
library(rpart) # 의사결정트리 기법 적용하기 위한 rpart 패키지 로드
library(e1071) # 나이브 베이즈 기법 적용하기 위한 e1071 패키지 로드
library(randomForest) # 랜덤 포레스트 기법 적용하기 위한 패키지 로드
rpart.iris <- rpart(Species~., data=iris_train) # 의사결정트리 머신러닝 적용
bayes.iris <- naiveBayes(Species~., data=iris_train) # 나이브 베이즈 머신러닝 적용 
rdf.iris <- randomForest(Species~., data=iris_train, importance=T) # 랜덤 포레스트적용
# 각 머신러닝 기법별로 예측 범주값 벡터 생성하기
rpart.pred <- predict(rpart.iris, newdata=iris_test, type="class")
bayes.pred <- predict(bayes.iris, newdata=iris_test, type="class")
rdf.pred <- predict(rdf.iris, newdata=iris_test, type="response")

table(iris_test$Species, rpart.pred) # 의사결정트리 적용 결과 혼동 행렬
table(iris_test$Species, bayes.pred) # 나이브 베이즈 적용 결과 혼동 행렬
table(iris_test$Species, rdf.pred) # 랜덤 포레스트 적용 결과 혼동 행렬

# 137p
confusionMatrix(rpart.pred, iris_test$Species, positive="versicolor")

# 138p
confusionMatrix(bayes.pred, iris_test$Species, positive="versicolor")

# 140p
confusionMatrix(rdf.pred, iris_test$Species, positive="versicolor")


"""