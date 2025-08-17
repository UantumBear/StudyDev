// frontend/src/pages/DevbearBotLinux.jsx
import React, { useEffect, useRef, useState } from "react";
import "./DevbearBotLinux.css";
import BearRunner from "../components/BearRunner";


/* 타자 속도(원하시면 숫자만 바꾸세요) */
const STATUS_TYPE_SPEED_MS   = 200; // "… 달려오고 있습니다 …" 한 글자 간격
const GREETING_TYPE_SPEED_MS = 80;  // "안녕하세요..! 제가 왔어요!" 한 글자 간격
const REPLY_TYPE_SPEED_MS    = 80;  // "응답입니다." 한 글자 간격

const GREETING_TEXT = "\n안녕하세요..! 제가 왔어요!";
const QUICK_REPLY   = "응답입니다.";

export default function DevbearBotLinux() {
  // 초기 상태 문구(흐리게)
  const [messages, setMessages] = useState([
    { role: "bot", type: "status", status: "home", text: "... 개발곰, 집에서 쉬는 중 ..." }
  ]);
  const [currentInput, setCurrentInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const [hasArrived, setHasArrived] = useState(false); // ★ 처음만 달려오게 하는 플래그
  const [showRunner, setShowRunner] = useState(false); // 달려오는 동안 곰 표시

  const termRef = useRef(null);
  const inputRef = useRef(null);
  const typingTimerRef = useRef(null);

  useEffect(() => {
    const focusInput = () => inputRef.current?.focus();
    focusInput();
    window.addEventListener("click", focusInput);
    return () => window.removeEventListener("click", focusInput);
  }, []);

  const scrollToBottom = () =>
    requestAnimationFrame(() =>
      termRef.current?.scrollTo(0, termRef.current.scrollHeight)
    );

  /* 1) 최초 1회: 달려오는 상태문구(느림) → 인사(빠름) */
  const startArrivalSequence = () => {
    setShowRunner(true); // 달려오는 곰 보이게 ㅏ기
    setIsTyping(true);

    // 상태 타자
    let i = 0;
    const statusMsg = "헛 둘 헛 둘, 개발곰이 달려오고 있습니다 ";
    setMessages(prev => [
        ...prev,
        { role: "bot", type: "status", status: "run", text: "" }
    ]);

    typingTimerRef.current = setInterval(() => {
      i++;
      setMessages(prev => {
        const list = [...prev];
        const last = list[list.length - 1];
        if (!last || last.type !== "status") return prev;
        last.text = statusMsg.slice(0, i);
        return list;
      });
      scrollToBottom();

      if (i >= statusMsg.length) {
        clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;

        // 인사 타자
        let j = 0;
        setMessages(prev => [...prev, { role: "botTyping", text: "" }]);
        typingTimerRef.current = setInterval(() => {
          j++;
          setMessages(prev => {
            const list = [...prev];
            const last = list[list.length - 1];
            if (!last || last.role !== "botTyping") return prev;
            last.text = GREETING_TEXT.slice(0, j);
            return list;
          });
          scrollToBottom();

          if (j >= GREETING_TEXT.length) {
            clearInterval(typingTimerRef.current);
            typingTimerRef.current = null;
            setMessages(prev => {
              const list = [...prev];
              list[list.length - 1] = { role: "bot", text: GREETING_TEXT };
              return list;
            });
            setIsTyping(false);
            setHasArrived(true); // ★ 이제부터는 빠른 응답만
            // setShowRunner(false); // 달려오는 곰 지우기
          }
        }, GREETING_TYPE_SPEED_MS);
      }
    }, STATUS_TYPE_SPEED_MS);


  };

  /* 2) 이후 매번: 간단한 “응답입니다.” 타자 */
  const startQuickReply = () => {
    setIsTyping(true);
    let k = 0;
    setMessages(prev => [...prev, { role: "botTyping", text: "" }]);
    typingTimerRef.current = setInterval(() => {
      k++;
      setMessages(prev => {
        const list = [...prev];
        const last = list[list.length - 1];
        if (!last || last.role !== "botTyping") return prev;
        last.text = QUICK_REPLY.slice(0, k);
        return list;
      });
      scrollToBottom();

      if (k >= QUICK_REPLY.length) {
        clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
        setMessages(prev => {
          const list = [...prev];
          list[list.length - 1] = { role: "bot", text: QUICK_REPLY };
          return list;
        });
        setIsTyping(false);
      }
    }, REPLY_TYPE_SPEED_MS);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      if (isComposing) return;
      e.preventDefault();
      if (isTyping) return;

      const text = currentInput.trim();
      if (!text) return;

      setMessages(prev => [...prev, { role: "user", text }]);
      setCurrentInput("");

      if (!hasArrived) startArrivalSequence();   // ★ 처음 1회만
      else startQuickReply();                    // ★ 이후에는 빠른 응답

      scrollToBottom();
    }
  };

  useEffect(() => {
    return () => typingTimerRef.current && clearInterval(typingTimerRef.current);
  }, []);

  return (
    <div className="linux-root">
      <div className="linux-titlebar">
        <div className="linux-title">DEVBEAR LLM CHATBOT</div>
      </div>

      <div className="linux-terminal" ref={termRef}>
        {messages.map((m, i) => (
          <div
            key={i}
            className={`linux-line ${m.role === "user" ? "right" : "left"} ${
              m.type === "status" ? "status" : ""
            }`}
          >
            {m.text}
            {/* 달려오는 status 줄일 때만 슬롯을 보유 → 슬롯은 항상 자리 차지 */}
            {m.type === "status" && m.status === "run" && (
              <span className="bear-slot">
                {showRunner ? (
                  <BearRunner scale={3} speed={120} color="#747373" />
                ) : null}
              </span>
            )}
          </div>
        ))}


        <div className="linux-line right">
          <input
            ref={inputRef}
            className="linux-inline-input"
            type="text"
            value={currentInput}
            onChange={(e) => setCurrentInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => setIsComposing(true)}
            onCompositionEnd={() => setIsComposing(false)}
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
          />
          <span className="linux-cursor">█</span>
        </div>
      </div>
    </div>
  );
}
