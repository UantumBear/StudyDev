tiktoken 라이브러리의 tiktoken_ext/openai_public.py 에는 아래와 같이 외부망을 바라보는 경로가 있다.
(https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken) 이런 식으로, 

###### Step 1.
폐쇄망에서 이용 시에는, 해당 위치를 참조하지 않도록, tiktoken 파일을 다운받은 후
아래와 같이 경로를 설정해준다. 
```python
def p50k_base():
    mergeable_ranks = load_tiktoken_bpe(
        # "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken", # 기존 
        # expected_hash="94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069", # 기존 
        f"{PROJECT_ROOT}/utilities/tiktoken/p50k_base.tiktoken" # 대체
    )
    return {
        "name": "p50k_base",
        "explicit_n_vocab": 50281,
        "pat_str": r50k_pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }
```

###### Step 2. 
처음에는 소스를 변경한 후, tiktoken 라이브러리를 tar.gz 파일로 추출하고, 해당 파일을 폐쇄망으로 반입하였는데,
설치가 되지 않았다.
tiktoken 라이브러리를 빌드하기 위해서는 rust 컴파일러가 필요했다.

rust 관련 설치 파일과 관련 의존성 파일을 전부 폐쇄망에 들여와서 설치하는 것은 방식도, 관리 측면에서도 힘들어 보였다.

때문에 폐쇄망에서는 간단히 일반 라이브러리들 처럼 whl 로 설치 할 수 있도록,
외부망에서 Docker Container 로 원하는 리눅스 환경을 만들고, 해당 환경에서 tiktoken-linux 용 .whl 라이브러리를 추출하는 방법을 택했다.

tiktoken 라이브러리 생성을 위한 Dockerfile은 아래와 같이 작성한다.
```docker
FROM python:3.12.0

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y build-essential

#  for tiktoken
RUN pip install setuptools_rust wheel
RUN pip install setuptools wheel
# rustup을 이용해 Rust 설치
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="/root/.cargo/bin:$PATH" && \
    rustc --version
# 환경변수 설정 (Rust 경로)
ENV PATH="/root/.cargo/bin:$PATH"

# 소스 코드 및 requirements.txt 복사
COPY thirdparty /app/thirdparty

# 작업 디렉토리 설정
WORKDIR /app/thirdparty/openai/tiktoken

# .whl 파일 생성
RUN python3 setup.py bdist_wheel

# .whl 파일을 /output 디렉토리로 이동
RUN mkdir /output && cp dist/*.whl /output/
```