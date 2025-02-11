## (PDF-1) 의사 결정 트리 분석 실습 ##

# install.packages("party") # 결정 트리(decision tree) 관련
library(party)
data(iris) # iris dataset 로드
samp <- c(sample(1:50, 25), sample(51:100,25), sample(101:150, 25))
# 1   ~  50 번 데이터 (Iris-setosa)     중 25개를 무작위 추출,
# 51  ~ 100 번 데이터 (Iris-versicolor) 중 25개를 무작위 추출,
# 101 ~ 150 번 데이터 (Iris-virginica)  중 25개를 무작위 추출.
# c() 는 단순한 1차월 벡터를 만드는 함수이다.
# 여러 개의 값들을 결합하여 하나의 벡터를 만든다.

# print(samp)


iris.tr <- iris[samp,]  # samp 에 속하는 인덱스들로 tr(training) data 를 만듦.
iris.te <- iris[-samp,] # samp 에 속하지 않는 인덱스들로 te(test) data 를 만듦.
iris_ctree <- ctree(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=iris.tr)
plot(iris_ctree) # 시각화


x <- subset(iris.te, select=-Species) # x: Species 는 목표변수 이므로 제외..
y <- iris.te$Species # y: 목표변수
DT_pred <- predict(iris_ctree, x) # 훈련된 결정 트리 모델 iris_ctree 에 x 를 집어넣어 예측을 수행. 
table(DT_pred, y)




## KNN ##
library(class) # class 클래스에는 KNN 알고리즘을 구현하는 knn() 함수가 있다.
data(iris)

print(iris)

y <- iris[,5] 
# 5번째 열(품종)을 추출하여 y 변수에 저장한다. 즉 예측하고자 하는 타깃은 '품종'이다.

tr.idx <- sample(length(y), 75) 
# y의 길이(150)에서 무작위로 75개의 인덱스를 선택하여 tr.idx 변수에 저장한다. 
# 이는 훈련 데이터를 선택하기 위함이다.

x.tr <- iris[tr.idx, -5] 
# tr.idx에 해당하는 행을 선택 + 5번째 열(타겟 변수)을 제외한 나머지를 훈련 데이터로 사용한다.

x.te <- iris[-tr.idx, -5] 
# tr.idx에 해당하지 않는 행을 선택 + 5번째 열(타겟 변수)은 제외한다.

m <- knn(x.tr, x.te, y[tr.idx], k=3) 
# knn() 함수를 사용하여 KNN 모델을 훈련하고 예측을 수행한다.
# 가장 가까운 3개의 이웃을 사용하여 분류를 수행한다.
# 이를 통해 m 에는 각 샘플에 대한 예측된 품종이 들어간다.

yy <- y[-tr.idx] 
# 테스트 데이터의 타겟 변수를 yy 에 저장한다.
table(yy,m)
# 예측된 품종(m)과 실제 품종(yy)을 비교하여 혼동 행렬(confusion matrix)을 생성한다,
# 이 결과를 통해 모델의 성능을 평가할 수 있다.



## Naive Bayes ## 
library(e1071) 
# 다양한 머신러닝 알고리즘을 포함하는 패키지

data(iris)

samp <- c(sample(1:50,35), sample(51:100,35), sample(101:150,35))
# 각 품종(setosa, versicolor, virginica)에서 35개씩 총 105개의 데이터를 랜덤 추출하기 위한 인덱스 추출
# samp 에는 인덱스가 들어간다.
print(samp)

iris.tr <- iris[samp,] 
# 선택된 index 를 기반으로 train set 구성 

iris.te <- iris[-samp,]
# 선택된 index 를 제외하고 test set 구성

m <- naiveBayes (Species ~ ., data = iris.tr)
# 붓꽃의 품종 (Species)을 목표 변수(종속 변수) 사용하여 나이브 베이즈 모델을 훈련한다.

pred <- predict(m, iris.te[,-5])
# test data set 에 대해 모델 m 을 사용하여 품종을 예측한다.

table(iris.te[,5], pred)
# 실제 품종(iris.te[,5])과 예측된 결과(pred)를 비교하여 혼동 행렬을 작성한다.



## SVM ##
library(e1071)  
# SVM과 나이브 베이즈를 포함하는 패키지

data(iris)

set.seed(123)  
# 난수 생성 과정의 시작점을 고정 (=시드 설정)
# 이렇게 하면, 코드를 실행할 때마다 동일한 랜덤 추출 결과를 얻을 수 있다.

tr.idx <- sample(1:nrow(iris), 0.7 * nrow(iris))  
# 훈련 데이터 인덱스 추출
# nrow(iris)는 iris 데이터셋의 행 수를 반환한다. 
#즉, 1부터 iris 데이터 전체 행 수 중 70% 를 랜덤 추출한 인덱스를 tr.idx 에 넣는다는 의미이다.

x.tr <- iris[tr.idx, -5]  # 훈련 데이터
x.te <- iris[-tr.idx, -5]  # 테스트 데이터
y.tr <- iris[tr.idx, 5]  # 훈련 데이터의 타겟
y.te <- iris[-tr.idx, 5]  # 테스트 데이터의 타겟

model_svm <- svm(Species ~ ., data = iris, subset = tr.idx)
# svm 알고리즘을 사용하여 모델을 훈련한다. data=iris 로 사용하며, 
# subset=tr.idx : tr.idx 에 해당하는 행 만을 사용하여 모델을 훈련시키겠다.

pred_svm <- predict(model_svm, x.te)
# 훈련된 모델(model_svm)을 사용하여 테스트 데이터(x.te)의 Species를 예측한다.

table(y.te, pred_svm)
# 실제 타겟 값(y.te)과 예측된 값(pred_svm)을 비교하는 혼동 행렬을 생성한다.

## Artificial Neural Network ##
library(neuralnet)
data(iris)

# iris 데이터의 타겟 변수를 One-hot Encoding 
iris$setosa <- ifelse(iris$Species == "setosa", 1, 0)
iris$versicolor <- ifelse(iris$Species == "versicolor", 1, 0)
iris$virginica <- ifelse(iris$Species == "virginica", 1, 0)
# 각 품종에 대해 별도의 열을 생성하고, 해당 품종에 맞는 경우 1로 설정하여
# 범주형 변수를 one hot 인코딩 하는 것이다.


# 훈련 및 테스트 데이터 분리
set.seed(123)
train_indices <- sample(1:nrow(iris), 0.7 * nrow(iris))  # 데이터의 70%를 훈련 데이터로 사용
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]

# 인공신경망 모델 구축
# 타겟 변수는 One-hot Encoding된 setosa, versicolor, virginica 3개 컬럼
nn_model <- neuralnet(setosa + versicolor + virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                      data = train_data, hidden = c(5), linear.output = FALSE)

# 신경망 구조 시각화
plot(nn_model)

# 테스트 데이터 준비
test_features <- subset(test_data, select = -c(Species, setosa, versicolor, virginica))

# 모델을 통한 예측 수행
nn_results <- compute(nn_model, test_features)
predictions <- nn_results$net.result

# 예측 결과를 통해 가장 높은 값을 가진 인덱스로 품종 예측
predicted_species <- apply(predictions, 1, which.max)
predicted_species <- factor(predicted_species, levels = c(1, 2, 3), labels = c("setosa", "versicolor", "virginica"))

# 실제 값과 예측 값 비교
confusion_matrix <- table(predicted_species, test_data$Species)
print(confusion_matrix)




# 랜덤 포레스트 관련 라이브러리 로드
install.packages("randomForest")  # 만약 설치되지 않았다면 설치


library(randomForest)
data(iris)  # iris 데이터셋 로드

# 훈련 및 테스트 데이터 분리
set.seed(123)
train_indices <- sample(1:nrow(iris), 0.7 * nrow(iris))  # 70%를 훈련 데이터로 사용
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]

# 랜덤 포레스트 모델 구축
rf_model <- randomForest(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                         data = train_data, ntree = 100)  # 100개의 트리 사용

# 모델 정보 출력
print(rf_model)

# 중요 변수 확인
importance(rf_model)
varImpPlot(rf_model)


# 테스트 데이터에 대한 예측 수행
rf_pred <- predict(rf_model, test_data)

# 실제 값과 예측 값 비교
confusion_matrix <- table(rf_pred, test_data$Species)
print(confusion_matrix)
