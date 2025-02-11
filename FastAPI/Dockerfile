# 공식 파이썬 이미지에서 3.12-slim 버전을 기반으로 함
FROM python:3.12.0
# 3.12.0 은 가능한데,3.12-slim 는 라이브러리간 호환성 충돌됨

# 컨테이너 내 디렉토리 설정
WORKDIR /admin


# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update --fix-missing && \
    apt-get install -y locales tzdata git vim curl

# 로케일 생성 및 설정
RUN sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

# 필요한 apt 리소스 설치
RUN apt-get -y install curl
RUN apt-get install -y libffi-dev
RUN apt-get install -y cmake
RUN apt-get install -y libcurl4-openssl-dev
RUN apt-get install -y unixodbc-dev
RUN apt-get install -y gpg
RUN apt-get install -y apt-transport-https
RUN apt-get update

# curl :명령줄 도구
# libffi-dev :Python 등 스크립트 언어
# cmake :컴파일
# libcurl4-openssl-dev :HTTP, FTP 등을 포함한 데이터 전송
# unixodbc-dev :Unix 시스템에서 ODBC (Open Database Connectivity) 지원
# gpg :데이터 암호화 및 서명
# apt-transport-https
# libpng-dev :PNG 이미지 처리
# update

# C:\Users\KB099\AppData\Local\Programs\Python\Python39\python.exe -m venv venv3910

# JDK 11 설치 (konlpy 호환 용도)
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk-headless  \
# 기존 11->17 로 바꿨음 문제 없나 확인하기.

# Rust 설치 (tiktoken 빌드에 필요)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 환경 변수 설정
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

# 환경 변수 설정
ENV LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko \
    LC_ALL=ko_KR.UTF-8

# 시간대 설정
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

# 패키지 정리 (도커 이미지 줄이기)
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 라이브러리 설치
# 폐쇄망의 경우, Nexus 를 통한 패키지 설치 설정 (로컬의 conf/pip.ini 를 참고, 넥서스에 없는 라이브러리는 로컬의 thirdparty 경로에서 가져옴)
# COPY conf/pip.ini /root/.pip/pip.conf
# COPY thirdparty /gpt-backend-kyobogpt-gptadmin/thirdparty

# 필요한 파일들 복사
# 의존성 파일을 컨테이너에 복사
COPY requirements_venv3120.txt .
# 따로 처리한 라이브러리 복사 (tiktoken, pykospacing)
# COPY thirdparty ./thirdparty/ # 집에서만 임시 주석
# 번역 파일 복사
# COPY translations ./translations/ # 집에서만 임시 주석


# pip 업그레이드 및 필요한 파이썬 패키지 설치
RUN python -m pip install --upgrade pip && \
    pip install -r requirements_venv3120.txt

# 번역 파일 권한 변경
# RUN chmod -R 777 translations


# 포트 설정 (외부에서 접근 가능한 포트들)
EXPOSE 80 8080 5003

# 기본 명령어 설정 (컨테이너가 실행될 때 기본적으로 수행될 명령어)
# FastAPI 애플리케이션 실행
CMD ["uvicorn", "run.app", "--host", "0.0.0.0", "--port", "5003"]


# 명령어
# docker build -t admin-i-base-001-312slim .
# docker run -d -p 5003:5003 --name gpt-backend-kyobogpt-gptadmin-c gpt-backend-kyobogpt-gptadmin-i-base-002


# 컨테이너는 실행하지 않고 이미지 접속하기.
# docker run -it gpt-backend-kyobogpt-gptadmin-i-base-002 /bin/bash
# 이미지 내 파일 확인하기.
# ls /usr/local/lib/python3.9/site-packages/pykospacing/resources/models/
# docker run -d -p 5003:5003 --name gpt-backend-kyobogpt-gptadmin-c-8 -v F:/kyobogpt/kyoboinsuranceGPT/gptadmApp:/gpt-backend-kyobogpt-gptadmin gpt-backend-kyobogpt-gptadmin-i-base-002
# # docker logs gpt-backend-kyobogpt-gptadmin-c-8

# 이미지 추출
# docker save -o gpt-backend-kyobogpt-gptadmin-image.tar gpt-backend-kyobogpt-gptadmin-i-base-002
