// @경로 FY2025LLM/frontend/src/pages/MainPage.js

import React from 'react'; //JSX 문법을 쓰기 위함
import './MainPage.css';
import { useNavigate } from 'react-router-dom'; // 훅(hook). 버튼 클릭 시 다른 페이지로 이동
import KakaoLoginButton from "../components/auth/KakaoLoginButton";


function MainPage() { // 이 파일이 정의하는 컴포넌트
  const navigate = useNavigate(); // 페이지 이동 기능을 담은 navigate 함수
  // return : 렌더링 할 화면
  return (
    <div className="main-container">
      <div className="title">UantumBear's Projects</div>


      {/* 🔹 레이아웃 래퍼: 왼쪽 소셜 스택 + 오른쪽 버튼들 */}
      <div className="hero-row">
        {/* 왼쪽: 소셜 로그인 스택 */}
        <aside className="social-panel" aria-label="소셜 로그인">
          <div className="social-stack">
            {/* 일반 로그인 버튼 */}
            <button
              className="devbear-login-button"
              onClick={() => navigate('/login')}
            >
              ˙u˙ Login
            </button>
            {/* 카카오 공식 축약형 로그인 버튼 */}
            <KakaoLoginButton />
            {/* <GoogleLoginButton /> */}
            {/* <GithubLoginButton /> */}
          </div>
        </aside>

        {/* 오른쪽: 프로젝트 버튼들 */}
        <div className="button-container">
          {/* 처음 만들었던, BERT 챗봇 */}
          <button
            className="pixel-button"
            onClick={() => window.location.href = "http://www.devbearbot.xyz/chat2"}
          >BERT Practice
          </button>
          {/* 두번째로 만들었던, 이력서 스타일의 웹 */}
          <button
            className="pixel-button"
            onClick={() => window.location.href = "http://www.devbearbot.xyz/resume/"}
          >RESUME
          </button>
          {/* LLM 개발곰 챗봇 → 게이트 경유 */}
          <button className="pixel-button" onClick={() => navigate('/gate/devbear')}>개발곰 LLM</button>

          {/* 내부: HR 챗봇 → 게이트 경유 */}
          <button
            className="pixel-button"
            aria-label="HR Chatbot Entry"
            onClick={() => navigate('/gate/hr')}
          >HR Chatbot
          </button>
        </div>

      </div>
      
    </div>
  );
}

export default MainPage;
