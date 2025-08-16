// src/pages/MainPage.js
import React from 'react';
import './MainPage.css';
import { useNavigate } from 'react-router-dom';
//

function MainPage() {
  const navigate = useNavigate();

  return (
    <div className="main-container">
      <div className="title">UantumBear's Projects</div>
      <div className="button-container">
        <button className="pixel-button" onClick={() => navigate('/devbear')}>개발곰 챗봇</button>
        <button className="pixel-button" onClick={() => navigate('/hr')}> HR Chatbot Entry</button>
      </div>
    </div>
  );
}

export default MainPage;
