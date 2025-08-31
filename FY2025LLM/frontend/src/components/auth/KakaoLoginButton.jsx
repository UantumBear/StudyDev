// @경로: frontend/src/components/auth/KakaoLoginButton.jsx
import React from "react";
import "./kakao.css";

export default function KakaoLoginButton() {
   // Vite 우선, 없으면 CRA, 그래도 없으면 기본값
  const API_BASE =
    (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL)
      || process.env.REACT_APP_API_BASE_URL
      || "http://localhost:8000";

  console.log("API_BASE =", API_BASE); // 이제 콘솔에 찍혀야 정상

  const handleClick = () => {
    window.location.href = `${API_BASE}/kakao/auth/login`;
  };

  return (
    <button
      type="button"
      className="kakao-login-short"
      onClick={handleClick}
      aria-label="카카오로 로그인"
    />
  );
}
