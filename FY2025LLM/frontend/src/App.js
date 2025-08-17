/*
 App.js
 @역할 실제 화면 UI와 라우팅 구조를 포함한 최상위 컴포넌트
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
/* import DevBearPage from './pages/DevBearPage'; */
import DevbearBotPage from './pages/DevbearBotPage';
import DevbearBotLinux from './pages/DevbearBotLinux';
import HrPage from './pages/HrPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        {/* <Route path="/devbear" element={<DevBearPage />} /> */}
        <Route path="/devbearBot" element={<DevbearBotPage />} />
        <Route path="/devbearBotLinux" element={<DevbearBotLinux />} />
        <Route path="/hr" element={<HrPage />} />
      </Routes>
    </Router>
  );
}

export default App;
