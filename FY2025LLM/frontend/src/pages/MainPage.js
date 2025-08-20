// frontend/src/pages/MainPage.js
import React from 'react'; //JSX 문법을 쓰기 위함
import './MainPage.css';
import { useNavigate } from 'react-router-dom'; // 훅(hook). 버튼 클릭 시 다른 페이지로 이동
//

function MainPage() { // 이 파일이 정의하는 컴포넌트
  const navigate = useNavigate(); // 페이지 이동 기능을 담은 navigate 함수
  // return : 렌더링 할 화면
  return (
    <div className="main-container">
      <div className="title">UantumBear's Projects</div>
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
        {/* HR 챗봇 */}
        <button className="pixel-button" onClick={() => navigate('/hr')}> HR Chatbot Entry</button>

      </div>
    </div>
  );
}

export default MainPage;
