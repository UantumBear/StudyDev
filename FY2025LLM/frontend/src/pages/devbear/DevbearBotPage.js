// frontend/src/pages/DevbearBotPage.jsx
import React, { useEffect, useRef, useState } from "react";
import "./DevbearBotPage.css";
const BOT_REPLY = "응답입니다.";

export default function DevbearBotPage() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "안녕하세요! 무엇이든 입력해 보세요 :)" },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const listRef = useRef(null);
  const typingTimerRef = useRef(null);

  // 스크롤 하단 고정
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // 타이핑 효과로 BOT_REPLY 출력
  const typeBotReply = async () => {
    setIsTyping(true);
    let idx = 0;

    // 우선 빈 메시지를 하나 추가해두고, 한 글자씩 채워 넣음
    const botIndex = messages.length + 1; // user 메시지 한 개가 먼저 들어갈 예정
    setMessages((prev) => [...prev, { role: "bot", text: "" }]);

    typingTimerRef.current = setInterval(() => {
      idx += 1;
      setMessages((prev) =>
        prev.map((m, i) =>
          i === botIndex ? { ...m, text: BOT_REPLY.slice(0, idx) } : m
        )
      );

      if (idx >= BOT_REPLY.length) {
        clearInterval(typingTimerRef.current);
        setIsTyping(false);
      }
    }, 90); // 타이핑 속도(밀리초) — 취향에 맞게 조절
  };

  const handleSend = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isTyping) return;

    // 유저 메시지 추가
    setMessages((prev) => [...prev, { role: "user", text }]);
    setInput("");

    // 봇 타이핑 시작
    typeBotReply();
  };

  // 언마운트/재렌더시 타이머 정리
  useEffect(() => {
    return () => typingTimerRef.current && clearInterval(typingTimerRef.current);
  }, []);

  return (
    <div className="dbb-root">
      <div className="dbb-frame">
        <div className="dbb-title">DevbearBot</div>

        <div className="dbb-chat" ref={listRef}>
          {messages.map((m, i) => (
            <div
              key={i}
              className={`dbb-msg ${m.role === "user" ? "me" : "bot"}`}
            >
              {m.text}
            </div>
          ))}
          {isTyping && (
            <div className="dbb-typing">▮</div> // 작은 커서 느낌
          )}
        </div>

        <form className="dbb-inputbar" onSubmit={handleSend}>
          <input
            className="dbb-input"
            type="text"
            placeholder="메시지를 입력하세요…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isTyping}
          />
          <button className="dbb-send" type="submit" disabled={isTyping}>
            {isTyping ? "응답 중…" : "보내기"}
          </button>
        </form>
      </div>
    </div>
  );
}
