// @경로 FY2025LLM/frontend/src/pages/hr/HrDtsBot.js

import React, { useState } from "react";
import "./HrDtsBot.css";

function HrDtsBot() {
  const [messages, setMessages] = useState([
    { role: "bot", content: "안녕하세요! HR 챗봇입니다. 무엇을 도와드릴까요?" }
  ]);
  const [input, setInput] = useState("");

  const sendMessage = () => {
    if (!input.trim()) return;
    setMessages([...messages, { role: "me", content: input }]);
    setInput("");
    // TODO: backend API 연동
    setTimeout(() => {
      setMessages(prev => [
        ...prev,
        { role: "bot", content: "현재는 데모 응답입니다. 😊" }
      ]);
    }, 500);
  };

  return (
    <div className="hrbot-container">
      <div className="hrbot-header">HR 챗봇</div>
      <div className="hrbot-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`hrbot-msg ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <div className="hrbot-input-area">
        <input
          className="hrbot-input"
          type="text"
          value={input}
          placeholder="메시지를 입력하세요..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button className="hrbot-send" onClick={sendMessage}>
          전송
        </button>
      </div>
    </div>
  );
}

export default HrDtsBot;
