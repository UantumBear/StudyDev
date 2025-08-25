// frontend/src/pages/DevbearGatePage.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './DevbearGatePage.css'; // ★ CSS 분리
const API_BASE = process.env.REACT_APP_API_BASE;
// CRA(Create React App)나 Vite 같은 빌드 도구는 process.env.REACT_APP_ 형태의 환경 변수를 빌드 시점에 JS 코드 안에 주입한다.
// 즉, npm start나 npm run build를 실행할 때, OS 환경 변수나 .env 파일에 정의된 값이 들어가는 것

function DevbearGatePage() {
  const navigate = useNavigate();
  const [key, setKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    setErr('');
    if (!key.trim()) {
      setErr('키를 입력해 주세요.');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/gate/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ service: 'devbear', key }),
      });
      const data = await res.json();
      if (!res.ok || !data?.ok) {
        throw new Error(data?.detail || '검증 실패');
      }
      sessionStorage.setItem('devbear_access_granted', 'true');
      navigate('/devbearBotLinux');
    } catch (e2) {
      setErr(e2.message || '알 수 없는 오류가 발생했어요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-container gate-center">
      <div className="gate-card">
        <h2 className="gate-title">접근 안내</h2>
        <p className="gate-desc">
          해당 페이지는 <b>유료 API</b>를 사용합니다. 접근을 위해 키를 입력해 주세요.
          <br />※ 실제 API 키가 아닌, 운영자가 정한 접근 키입니다.
        </p>
        <form onSubmit={submit}>
          <label htmlFor="gatekey" className="gate-label">접근 키</label>
          <input
            id="gatekey"
            type="password"
            className="gate-input"
            placeholder="키를 입력하세요"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            autoFocus
          />
          {err && <div className="gate-error">{err}</div>}
          {/* error mesage 렌더링 */}
          <button
            type="submit"
            className="pixel-button gate-submit"
            disabled={loading}
          >
            {loading ? '확인 중...' : '확인하고 들어가기'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default DevbearGatePage;
