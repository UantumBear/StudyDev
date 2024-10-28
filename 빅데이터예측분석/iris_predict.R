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
table(DT_pred, dt_y)

## (PDF-2) KNN ##
library(class)
data(iris)

y <- iris[,5]
tr.idx <- sample(length(y), 75)
x.tr <- iris[tr.idx, -5]
x.te <- iris[-tr.idx, -5]
m <- knn(x.tr, x.te, y[tr.idx], k=3)
yy <- y[-tr.idx]
table(yy,m)

## (PDF-3) Naive Bayes ## 
library(e1071)
data(iris)
samp <- c(sample(1:50,35), sample(51:100,35), sample(101:150,35))
iris.tr <- iris[samp,]
iris.te <- iris[-samp,]
m <- naiveBayes (Species ~ ., data = iris.tr)
pred <- predict(m, iris.te[,-5])
table(iris.te[,5], pred)

