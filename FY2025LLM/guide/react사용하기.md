### 프로젝트 셋팅 방법

#### Node.js와 npm 설치
https://nodejs.org/ko 에서 LTS 버전 다운로드  
Windows 64-bit Installer: https://nodejs.org/dist/v22.17.1/node-v22.17.1-x64.msi  

| 설치해야할 파일 | node-v22.17.1-x64.msi |
|----------|-----------------------|

`Node.js`는 해석기(런타임) 역할로, 시스템에 설치하는 것이다.  
Node.js는 시스템에 한 번 설치해두고,  
프로젝트 별로 npm install을 통해 의존성을 관리한다.  
`npm`은 Node.js의 공식 패키지 매니저이다. node js 를 설치하면 자동으로 설치된다.  

###
```shell
cd C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM\frontend

# 의존성 설치 (package.json에 정의된 라이브러리를 node_modules에 설치한다.)
npm install
```

###### 설치 확인
```shell
node -v
npm -v

# 결과:
v22.17.1
10.9.2
```

##### npm install
npm install과 npm run dev 같은 명령어는 package.json이 있는 폴더에서 실행해야 한다. 
```shell
cd frontend
# frontend 경로에서 아래 명령어 수행
npm install
npm install react-router-dom
npm run dev
# npm run dev 란, package.json의 "scripts" 항목에서 "dev" 키에 등록된 명령을 실행하는 것
```


##### 기존 jinja templates 방식과 react 차이 비교
FastAPI 에서 templates/ 는 서버에서 HTML을 생성해서 사용자에게 렌더링하여 반환.  
React 에서 public 은 딱 한번 로딩되는 HTML 툴,  
React 는 이 안의 <div id="root"> 안에, js 로 UI를 그려 넣음.  
HTML 에 데이터를 넣지 않고, 모든 UI를 JS(JSX) 를 통해 그림..  


