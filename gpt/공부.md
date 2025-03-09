### 파이프라인이란?

파이프라인이란, 데이터를 처리하는 여러 단계를 자동화하여 연결하는 개념이다.
하나의 작업이 끝나면, 그 결과를 다음 작업으로 넘겨주는 방식을 말한다.

##### 파이프라인이 사용되는 분야
- 데이터 처리
- 머신러닝 (모델 학습, 배포)
- 소프트웨어 배포 (CI/CD 파이프라인)
- API 요청 처리

##### 예시 코드
```python
# 머신러닝 학습 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. 데이터 전처리 (스케일링)
# 2. 모델 학습
ml_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression())
])

ml_pipeline.fit(X_train, y_train)  # 전체 파이프라인 실행
```

#### 머신러닝 분야에서 파이프라인 사용 방식
사용자가 .py 파일을 업로드하면,
AI 모델을 추가하거나, 데이터 처리 파이프라인을 변경함.
업로드된 .py 파일을 특정 디렉터리에 저장한 후 실행해서 동적으로 파이프라인을 로드.

왜 .py 파일을 업로드할까?
##### AI 모델을 동적으로 업데이트 하고 싶기 때문.
AI 모델을 계속 학습하고 개선해야 하는 경우,  
모델이 포함된  .py 파일을 업로드하면 시스템을 다시 배포하지 않고도,  
새로운 모델을 적용할 수 있기 때문이다.

##### 데이터 전처리 방식이 계속 바뀔 때
머신러닝에서는 새로운 전처리 방식이 필요할 때가 많다.  
예를 들어 normalize 하는 방식이 바뀌면, 새 .py 파일을 올려서 바로 적용한다.

```python
# 예를 들면,
# 기존 방식
def preprocess(data):
    return data / 100

# 새로운 방식 (업로드된 .py 파일)
def preprocess(data):
    return (data - min(data)) / (max(data) - min(data))
```

#### Chatbot Frameworks 예서 파이프라인을 사용하는 방식

Rasa 와 같은 AI 챗봇 프레임워크에서는 .py 파일을 업로드하여  
챗봇의 동작을 변경할 수 있다고 한다.  
즉, 대화 흐름 (NLU, Dialog Management) 나,  
RAG Agent 의 처리방식을 동적으로 업데이트 하는 것이다.  

RAG 기반 AI 챗봇에서는, 
대화 흐름(Dialog Management), 검색 방식(Retrieval), AI 모델 적용 방식을   
.py 파일로 관리 할 수 있다.

###### Q.  
그럼..  
사용자의 질문을 기반으로  
첫번째 API 를 받는 라우터가  
어떤 파이프라인을 사용할 지 (단순 검색 , RAG 등)를 결정하고, 
RAG agent 서버에 특정 py 파일을 선택하여 업로드를 요청하고..
그 파일을 실행해서 질문을 처리하고 응답을 반환하는  
그런 느낌인가..?