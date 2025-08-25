// @경로 FY2025LLM/frontend/src/pages/hr/HrGatePage.js

import React from "react";
import { useNavigate } from "react-router-dom";
import "./HrGatePage.css";

function HrGatePage() {
  const navigate = useNavigate();

  const enterHr = () => {
    sessionStorage.setItem("hr_access_granted", "true"); // ★추가
    navigate("/hrbot");
  };

  return (
    <div className="hrgate-container">
      <div className="hrgate-card">
        <h1 className="hrgate-title">HR 챗봇</h1>
        <p className="hrgate-subtitle">인사 관련 질문을 편리하게 해결하세요.</p>
        <button className="hrgate-button" onClick={enterHr}>
          시작하기
        </button>
      </div>
    </div>
  );
}

export default HrGatePage;
