# from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Callable

class KeywordVectorizer:
    def __init__(self, tokenizer: Callable[[str], list[str]] = None, max_keywords: int = 5):
        self.tokenizer = tokenizer or self.default_tokenizer
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, max_features=50)
        self.max_keywords = max_keywords

    """ 토크 나이저 함수 정의 """

    def default_tokenizer(self, text: str) -> list[str]:
        # 띄어쓰기 기준으로 나눔 (간단한 한글 분석기 대체)
        return text.split()

    # def okt_tokenize(self, text: str):
    #     return self.okt.nouns(text)

    """ 기능 """
    def extract_tags(self, texts: list[str]) -> list[list[str]]:
        self.vectorizer.fit(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        result = []
        for text in texts:
            tfidf_vector = self.vectorizer.transform([text]).toarray()[0]
            top_indices = tfidf_vector.argsort()[::-1][:self.max_keywords]
            top_words = [feature_names[i] for i in top_indices if tfidf_vector[i] > 0]
            result.append(top_words)
        return result


if __name__ == "__main__":
    # 테스트용 문장 리스트
    texts = [
        "개발곰은 오늘도 새벽까지 버그와 사투를 벌이며 코드를 수정하고 있었다.",
        "깃허브에 첫 오픈소스를 등록한 날, 개발곰은 세상 누구보다 뿌듯한 미소를 지었다.",
        "커피 한 잔을 내려 마신 개발곰은 딥러닝 모델의 성능이 갑자기 향상된 이유를 추적하기 시작했다.",
        "사막 같은 로그를 헤매던 개발곰은 마침내 원인을 찾아내고 외쳤다. '이거였구나!'",
        "오랜만에 자연어 처리 프로젝트를 맡게 된 개발곰은 konlpy와 함께 단어를 분석하며 하루를 보냈다.",
        "개발곰은 개발곰이라는 단어를 좋아했다."
    ]

    vectorizer = KeywordVectorizer(max_keywords=5)

    # def okt_tokenizer(text: str) -> list[str]:
    #     return Okt().nouns(text)
    # vectorizer = KeywordVectorizer(tokenizer=okt_tokenizer, max_keywords=5)

    tag_results = vectorizer.extract_tags(texts)

    for i, tags in enumerate(tag_results):
        print(f"[문장 {i + 1}]")
        print("원문:", texts[i])
        print("추출된 키워드:", tags)
        print("-" * 40)


# python utils/nlp/vectorizer.py