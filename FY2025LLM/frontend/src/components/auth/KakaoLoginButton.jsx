// @경로: frontend/src/components/auth/KakaoLoginButton.jsx
import React from "react";
import "./kakao.css";

export default function KakaoLoginButton({ redirectTo }) {
  const API_BASE = process.env.REACT_APP_API_BASE_URL || "";
  const handleClick = () => {
    const redirect = redirectTo || `${window.location.origin}/auth/kakao/callback`;
    const url = `${API_BASE}/auth/kakao/login?redirect_uri=${encodeURIComponent(redirect)}`;
    window.location.href = url;
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
