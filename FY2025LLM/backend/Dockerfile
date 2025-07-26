# 1. Python 베이스 이미지 선택
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 로컬 코드 복사
COPY . .

# 4. 필요한 패키지 설치
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements/requirements.txt

# 5. FastAPI 서버 실행 명령 (reload는 개발 시에만 사용)
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
