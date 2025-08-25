/*
 @경로 FY2025LLM/frontend/src/App.js
 @역할 실제 화면 UI와 라우팅 구조를 포함한 최상위 컴포넌트
 */
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import MainPage from './pages/MainPage';

// DevBear
// import DevbearBotPage from './pages/devbear/DevbearBotPage';
import DevbearGatePage from './pages/devbear/DevbearGatePage'; /* Linux 접속 전 대문 페이지 */
import DevbearBotLinux from './pages/devbear/DevbearBotLinux';

// HR
import HrGatePage from './pages/hr/HrGatePage'; /* HR 접속 전 대문 페이지 */
import HrDtsBot from './pages/hr/HrDtsBot';

function ThemeController() {
  const location = useLocation();

  useEffect(() => {
    const path = location.pathname;

    const isDevbear =
      path.startsWith('/gate/devbear') ||
      path.startsWith('/devbear') ||       // /devbearBot, /devbearBotLinux 등 포함
      path.includes('devbear');

    const isHr =
      path.startsWith('/gate/hr') ||
      path.startsWith('/hr');

    // 초기화
    document.body.classList.remove('devbear-theme', 'hr-theme');

    if (isDevbear) document.body.classList.add('devbear-theme');
    else if (isHr) document.body.classList.add('hr-theme');
  }, [location.pathname]);

  return null;
}

// ★ 보호 라우트: 세션에 통과 플래그 없으면 게이트로 보냄
// function ProtectedRoute({ children }) {
//   const granted = sessionStorage.getItem('devbear_access_granted') === 'true';
//   return granted ? children : <Navigate to="/gate/devbear" replace />;
// }
// ★ 보호 라우트: 세션 키와 리다이렉트 경로를 받도록 일반화
function ProtectedRoute({ children, keyName, redirectPath }) {
  const granted = sessionStorage.getItem(keyName) === 'true';
  return granted ? children : <Navigate to={redirectPath} replace />;
}



function App() {
  return (
    <Router>
      {/* 서비스별 테마용 ThemeController 클래스 토글러 마운트 */}
      <ThemeController />


      <Routes>
        {/* 메인 */}
        <Route path="/" element={<MainPage />} />

        {/* DevBear */}
        {/* <Route path="/devbear" element={<DevBearPage />} /> */}
        {/* <Route path="/devbearBot" element={<DevbearBotPage />} /> */}
        <Route path="/gate/devbear" element={<DevbearGatePage />} />
        {/* ★ 보호 라우트로 감싸기 */}
        <Route
          path="/devbearBotLinux"
          element={
            <ProtectedRoute keyName="devbear_access_granted" redirectPath="/gate/devbear">
              <DevbearBotLinux />
            </ProtectedRoute>
          }
        />
        {/* HR */}
        <Route path="/gate/hr" element={<HrGatePage />} />
        <Route
          path="/hrbot"
          element={
            <ProtectedRoute keyName="hr_access_granted" redirectPath="/gate/hr">
              <HrDtsBot />
            </ProtectedRoute>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
