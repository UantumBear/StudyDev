// frontend/src/App.jsx
import { useState } from "react";
import axios from "axios";

function App() {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);

  const sendMessage = async () => {
    const res = await axios.post("http://localhost:8000/chat", { message: input });
    setChat([...chat, { role: "user", text: input }, { role: "bot", text: res.data.reply }]);
    setInput("");
  };

  return (
    <div>
      <div>
        {chat.map((msg, i) => (
          <div key={i} style={{ textAlign: msg.role === "user" ? "right" : "left" }}>
            {msg.text}
          </div>
        ))}
      </div>
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={sendMessage}>전송</button>
    </div>
  );
}

export default App;
