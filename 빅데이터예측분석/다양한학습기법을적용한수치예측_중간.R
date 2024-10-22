# 필요한 패키지 설치하기
# install.packages("MASS")
# install.packages("tree")
# install.packages("rpart.plot")
# install.packages("neuralnet")
# install.packages("randomForest")
# install.packages("caret")
# install.packages("arules")
# install.packages("arulesViz")

library(MASS) # MASS 패키지 로드
data(Boston) # Boston dataset 로드
# CSV 파일로 저장 (파이썬에서 동일한 데이터셋 이용하기 위함.)
write.csv(Boston, file = "C:/devbear/WorkSpaces/DevBear/manual/sogang/빅데이터예측분석/중간과제/boston_data.csv", row.names = FALSE) 

# 69p
idx <- sample(1:nrow(Boston), size=nrow(Boston)*0.7, replace=F) 
# Boston 데이터의 70%를 랜덤으로 선택하여 훈련 데이터 인덱스를 생성, replace=F:중복선택 미허용.

Boston_train <- Boston[idx, ] # 훈련용 데이터 세트 분할 (idx 를 사용해 Boston 데이터에서 해당 인덱스의 값들을 추출)
Boston_test <- Boston[-idx, ] # 평가용 데이터 세트 분할 (idx 포함되지 않은 데이터들을 추출)
dim(Boston_train) ; dim(Boston_test) # 데이터 프레임의 행과 열의 개수를 반환

lm.fit <- lm(medv~., data=Boston_train) # 모든 독립 변수를 사용하여 선형 회귀 모델을 적합
summary(lm.fit)

# 70p
lm.fit2 <- step(lm.fit, method="both") # 변수 선택을 자동으로 수행하여 더 나은 모델을 찾음
summary(lm.fit2)

# 71p
lm.yhat2 <- predict(lm.fit2, newdata=Boston_test) # Boston_test 데이터를 입력으로 받아, 예측값 생성
mean((lm.yhat2-Boston_test$medv)^2) # 예측값과 실제값간의 평균제곱오차(MSE) 계산

# 72p
library(tree)
tree.fit <- tree(medv~., data=Boston_train)
summary(tree.fit)

## Error: > plot(tree.fit)
## plot.new()에서 다음과 같은 에러가 발생했습니다: figure margins too large
## 원인: R studio 에서 실제 plots 창 크기 때문이었다.
par("mar") # 현재 설정된 마진 확인
par("pin") # 그래픽 창 크기 확인  
## 해결방안:
windows()  # 새로운 그래픽 창 열기

plot(tree.fit)
text(tree.fit, pretty=0) # pretty=0: 기본값을 사용하겠다.

# 73p
tree.yhat <- predict(tree.fit, newdata=Boston_test) # 평가 데이터 이용, 예측결과생성
mean((tree.yhat-Boston_test$medv)^2) # 예측값과 실제값간 평균제곱오타(MSE) 계산

library(rpart)
rpart.fit <-rpart(medv~., data=Boston_train)
summary(rpart.fit)

# 74p
library(rpart.plot)
rpart.plot(rpart.fit, digits=3, type=0, extra=1, fallen.leaves=F, cex=1)

# 75p
rpart.yhat <- predict(rpart.fit, newdata=Boston_test) # 평가 데이터 이용, 예측결과생성
mean((rpart.yhat-Boston_test$medv)^2) # 예측 값과 실제 값간 평균제곱오차(MSE) 계산

## 정규화 함수 작성하기
normalize <- function(x){ return((x-min(x))/(max(x)-min(x))) }
Boston_train_norm <- as.data.frame(sapply(Boston_train, normalize))
Boston_test_norm <- as.data.frame(sapply(Boston_test, normalize))

# (1) nnet 함수를 사용한 인공 신경망 분석
library(nnet)
nnet.fit <- nnet(medv~., data=Boston_train_norm, size=5) # 인공 신경망 적합하기
nnet.yhat <- predict(nnet.fit, newdata=Boston_test_norm, type="raw") # 예측결과생성
mean((nnet.yhat-Boston_test_norm$medv)^2) # 평균제곱오차 계산

# 76p
library(neuralnet)
neural.fit <- neuralnet(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio
                        +black+lstat, data=Boston_train_norm, hidden=5) # 인공 신경망 적합하기
neural.results <- compute(neural.fit, Boston_test_norm[1:13]) # 예측결과생성
neural.yhat <- neural.results$net.result
mean((neural.yhat-Boston_test_norm$medv)^2) # 평균제곱오차 계산

# 77p
library(randomForest)
set.seed(1)
rf.fit <- randomForest(medv~., data=Boston_train, mtry=6, importance=T)
rf.fit

# 78p
importance(rf.fit)
varImpPlot(rf.fit)

# 79p
rf.yhat <- predict(rf.fit, newdata=Boston_test) # 평가 데이터 이용, 예측결과생성
mean((rf.yhat-Boston_test$medv)^2) # 예측값과 실제값간 평균제곱오차(MSE) 계산


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



