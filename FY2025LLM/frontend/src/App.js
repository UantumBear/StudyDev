/*
 App.js
 @역할 실제 화면 UI와 라우팅 구조를 포함한 최상위 컴포넌트
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
/* import DevBearPage from './pages/DevBearPage'; */
import DevbearBotPage from './pages/DevbearBotPage';

import DevbearGatePage from './pages/DevbearGatePage'; /* Linux 접속 전 대문 페이지 */
import DevbearBotLinux from './pages/DevbearBotLinux';
import HrPage from './pages/HrPage';

// ★ 보호 라우트: 세션에 통과 플래그 없으면 게이트로 보냄
function ProtectedRoute({ children }) {
  const granted = sessionStorage.getItem('devbear_access_granted') === 'true';
  return granted ? children : <Navigate to="/gate/devbear" replace />;
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        {/* <Route path="/devbear" element={<DevBearPage />} /> */}
        <Route path="/devbearBot" element={<DevbearBotPage />} />
        <Route path="/gate/devbear" element={<DevbearGatePage />} />
        {/* ★ 보호 라우트로 감싸기 */}
        <Route
          path="/devbearBotLinux"
          element={
            <ProtectedRoute>
              <DevbearBotLinux />
            </ProtectedRoute>
          }
        />
        <Route path="/hr" element={<HrPage />} />
      </Routes>
    </Router>
  );
}

export default App;
