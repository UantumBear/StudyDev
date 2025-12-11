# pip install ragas openai python-dotenv datasets
# C:\Users\KB099\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
# C:\Users\KB099\AppData\Local\Programs\Python\Python312\python.exe -m pip install --upgrade pip
# 참고 논문 RAGAS: Automated Evaluation of Retrieval Augmented Generation
# @실행명령어: python LLM/rag/ragas_test.py
# 환경변수 설정: .env 파일에 OPENAI_API_KEY=your_openai_api_key_here

import os
from dotenv import load_dotenv
import random
import pandas as pd

# .env 파일 로드
load_dotenv()

from ragas.metrics import (
    ContextRelevance,
    AnswerRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)
from ragas import evaluate
from datasets import Dataset

print("=== RAGAS OpenAI 평가 시스템 ===")

# OpenAI API 키 확인
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    print()
    print("설정 방법:")
    print("1. .env 파일에 추가: OPENAI_API_KEY=your_openai_api_key")
    print("2. 환경변수로 설정: set OPENAI_API_KEY=your_openai_api_key")
    print("3. OpenAI API 키 발급: https://platform.openai.com/api-keys")
    print()
    print("OpenAI API는 $5 정도면 충분히 테스트 가능합니다.")
    
    # 더미 평가 실행
    print("\n" + "="*50)
    print("더미 평가 데이터로 코드 구조 테스트")
    print("="*50)
    
    # 샘플 데이터
    sample_data = {
        "question": [
            "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
            "머신러닝에서 overfitting이란 무엇인가요?",
            "FastAPI의 주요 장점은 무엇인가요?"
        ],
        "answer": [
            "리스트는 변경 가능(mutable)하고 대괄호[]를 사용하며, 튜플은 변경 불가능(immutable)하고 소괄호()를 사용합니다.",
            "Overfitting은 모델이 훈련 데이터에 너무 특화되어 새로운 데이터에서 성능이 떨어지는 현상입니다.",
            "FastAPI는 빠른 성능, 자동 API 문서 생성, 타입 힌트 지원 등이 주요 장점입니다."
        ],
        "contexts": [
            [
                "Python의 리스트는 mutable(변경 가능한) 데이터 타입입니다.",
                "튜플은 immutable(변경 불가능한) 데이터 타입으로 생성 후 수정할 수 없습니다.",
                "리스트는 대괄호[]로 선언하고, 튜플은 소괄호()로 선언합니다."
            ],
            [
                "Overfitting은 기계학습에서 중요한 문제 중 하나입니다.",
                "훈련 데이터에 과도하게 맞춰져서 일반화 성능이 떨어지는 현상입니다.",
                "검증 데이터에서 성능이 훈련 데이터보다 현저히 떨어질 때 의심해볼 수 있습니다."
            ],
            [
                "FastAPI는 Python 기반의 현대적인 웹 프레임워크입니다.",
                "Starlette과 Pydantic을 기반으로 구축되어 뛰어난 성능을 제공합니다.",
                "자동으로 OpenAPI 스펙을 생성하여 API 문서를 제공합니다."
            ]
        ],
        "ground_truth": [
            "리스트는 변경 가능하고 튜플은 변경 불가능한 Python 데이터 구조입니다.",
            "Overfitting은 모델이 훈련 데이터에 과적합되어 새 데이터에서 성능이 저하되는 문제입니다.",
            "FastAPI는 고성능, 자동 문서화, 타입 안전성을 제공하는 Python 웹 프레임워크입니다."
        ]
    }
    
    # 더미 평가 결과 생성
    dummy_results = {
        'user_input': sample_data['question'],
        'response': sample_data['answer'],
        'retrieved_contexts': sample_data['contexts'],
        'reference': sample_data['ground_truth'],
        'context_relevance': [round(random.uniform(0.7, 0.9), 4) for _ in range(3)],
        'answer_relevancy': [round(random.uniform(0.6, 0.85), 4) for _ in range(3)],
        'faithfulness': [round(random.uniform(0.75, 0.95), 4) for _ in range(3)]
    }
    
    print("\n=== 더미 평가 결과 (참고용) ===")
    df = pd.DataFrame(dummy_results)
    print(df[['user_input', 'context_relevance', 'answer_relevancy', 'faithfulness']].to_string())
    print()
    
    # 메트릭별 평균 점수
    for metric in ['context_relevance', 'answer_relevancy', 'faithfulness']:
        avg_score = df[metric].mean()
        print(f"{metric}: {avg_score:.4f}")
    
    overall_avg = df[['context_relevance', 'answer_relevancy', 'faithfulness']].mean().mean()
    print(f"\n전체 평균 점수: {overall_avg:.4f}")
    
    print("\n" + "="*50)
    print("OpenAI API 키를 설정하고 다시 실행하면 실제 평가가 진행됩니다!")
    
    exit()

print("OpenAI API 키가 설정되었습니다.")

# 평가용 메트릭 정의 (기본 OpenAI 사용)
metrics = [
    ContextRelevance(),
    AnswerRelevancy(), 
    Faithfulness(),
]

print("OpenAI를 사용한 평가 메트릭이 설정되었습니다.")

# 샘플 데이터 - RAG 시스템의 결과물 예시
sample_data = {
    "question": [
        "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
        "머신러닝에서 overfitting이란 무엇인가요?",
        "FastAPI의 주요 장점은 무엇인가요?"
    ],
    "answer": [
        "리스트는 변경 가능(mutable)하고 대괄호[]를 사용하며, 튜플은 변경 불가능(immutable)하고 소괄호()를 사용합니다.",
        "Overfitting은 모델이 훈련 데이터에 너무 특화되어 새로운 데이터에서 성능이 떨어지는 현상입니다.",
        "FastAPI는 빠른 성능, 자동 API 문서 생성, 타입 힌트 지원 등이 주요 장점입니다."
    ],
    "contexts": [
        [
            "Python의 리스트는 mutable(변경 가능한) 데이터 타입입니다.",
            "튜플은 immutable(변경 불가능한) 데이터 타입으로 생성 후 수정할 수 없습니다.",
            "리스트는 대괄호[]로 선언하고, 튜플은 소괄호()로 선언합니다."
        ],
        [
            "Overfitting은 기계학습에서 중요한 문제 중 하나입니다.",
            "훈련 데이터에 과도하게 맞춰져서 일반화 성능이 떨어지는 현상입니다.",
            "검증 데이터에서 성능이 훈련 데이터보다 현저히 떨어질 때 의심해볼 수 있습니다."
        ],
        [
            "FastAPI는 Python 기반의 현대적인 웹 프레임워크입니다.",
            "Starlette과 Pydantic을 기반으로 구축되어 뛰어난 성능을 제공합니다.",
            "자동으로 OpenAPI 스펙을 생성하여 API 문서를 제공합니다."
        ]
    ],
    "ground_truth": [
        "리스트는 변경 가능하고 튜플은 변경 불가능한 Python 데이터 구조입니다.",
        "Overfitting은 모델이 훈련 데이터에 과적합되어 새 데이터에서 성능이 저하되는 문제입니다.",
        "FastAPI는 고성능, 자동 문서화, 타입 안전성을 제공하는 Python 웹 프레임워크입니다."
    ]
}

# Dataset 생성
dataset = Dataset.from_dict(sample_data)

print("=== RAGAS 평가 시작 ===")
print(f"평가할 샘플 수: {len(dataset)}")
print(f"사용 메트릭: {[metric.__class__.__name__ for metric in metrics]}")
print()

# 평가 실행
try:
    result = evaluate(
        dataset,
        metrics=metrics,
    )
    
    print("=== 평가 결과 ===")
    if hasattr(result, 'to_pandas'):
        # ragas 최신 버전의 결과 처리
        df = result.to_pandas()
        print(df)
        
        # 각 메트릭 평균 계산
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            avg_score = df[col].mean()
            print(f"{col}: {avg_score:.4f}")
            
        overall_avg = df[numeric_cols].mean().mean()
        print(f"\n전체 평균 점수: {overall_avg:.4f}")
            
    else:
        # 이전 버전 호환성
        for metric_name, score in result.items():
            print(f"{metric_name}: {score:.4f}")
            
        print(f"\n전체 평균 점수: {sum(result.values())/len(result):.4f}")
        
    print("\nOpenAI를 사용한 RAGAS 평가가 완료되었습니다!")
    
except Exception as e:
    print(f"평가 중 오류 발생: {e}")
    print("OpenAI API 키가 올바른지, 잔액이 충분한지 확인해주세요.")