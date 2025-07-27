# frontend/Dockerfile

# 1. Node.js base image 사용
FROM node:20

# 2. 앱 폴더 설정
WORKDIR /FY2025LLM/frontend

# 3. package.json, package-lock.json 복사
COPY package*.json ./

# 4. 의존성 설치
RUN npm install

# 5. 나머지 소스 복사
COPY . .

# 6. 개발 서버 실행
CMD ["npm", "run", "start"]
