import { useState, useRef, useEffect } from "react";
import { Send, Bot } from "lucide-react";

const getUserId = () => {
  let userId = "444";
  // let userId = localStorage.getItem("chatUserId");
  if (!userId) {
    // userId = crypto.randomUUID();
    userId = "444";
    localStorage.setItem("chatUserId", userId);
  }
  return userId;
};

const fetchResponse = async (message) => {
  const userId = getUserId();
  console.log("before fetch");
  const trimmed = message.trim();
  console.log(trimmed);
  const response = await fetch(
    `https://projektetykabe.onrender.com/api/openai/chat?userId=${encodeURIComponent(
      userId
    )}&prompt=${encodeURIComponent(message.trim())}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    }
  );

  console.log("HTTP status:", response.status);

  if (!response.ok) {
    console.log("BAD RESPONSE");
    throw new Error("Network response was not ok");
  }
  const data = await response.text();
  console.log(data);
  return data;
};

export default function ChatApp() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Witaj! Jak Ci mogę dzisiaj pomóc?" },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input.trim() };
    console.log(input.trim());

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    console.log("BEFORE TRY");

    try {
      const responseText = await fetchResponse(input.trim());
      console.log(`RESPONSE: ${responseText}`);

      let contentToShow = responseText;
      if (responseText === "Prompt spoza zakresu aplikacji.") {
        contentToShow =
          "Niestety, nie mogę odpowiedzieć na to pytanie, ponieważ jest poza zakresem mojej aplikacji.";
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: contentToShow },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Przepraszam, wystąpił błąd podczas przetwarzania zapytania.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Asystent anonimizacji danych</h1>
      </header>

      <div className="messages-container">
        {messages.length === 1 && (
          <div className="initial-message">
            <div className="bot-icon">
              <Bot size={24} />
            </div>
            <p>{messages[0].content}</p>
          </div>
        )}

        <div className="message-list">
          {messages.length > 1 &&
            messages.map((message, i) => (
              <div
                key={i}
                className={`message ${
                  message.role === "user" ? "user" : "assistant"
                }`}
              >
                {message.content}
              </div>
            ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <form className="input-area" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            ref={inputRef}
            autoComplete="off"
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            <Send size={20} />
          </button>
        </div>
      </form>
    </div>
  );
}
