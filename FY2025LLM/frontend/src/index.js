/*
frontend/src/index.js
애플리케이션의 진입점. HTML의 <div id="root">에 React 앱을 연결하는 역할
*/

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
