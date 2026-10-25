import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { MAX_MESSAGE_LENGTH, sendMessage } from "./api";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
}

type Status = "idle" | "loading" | "blocked" | "rate_limited" | "error";

const EXAMPLES = [
  "What is the current weather in Tokyo?",
  "What was the PM2.5 level in Beijing in your historical database?",
];

const NOTICES: Record<"blocked" | "rate_limited" | "error", string> = {
  blocked: "That message was blocked by the safety checks. Please rephrase your question.",
  rate_limited: "You're sending messages too quickly. Please wait a minute and try again.",
  error: "Something went wrong. Please try again.",
};

let nextId = 0;

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  // The message to resend if the user clicks Retry after an error.
  const [failedText, setFailedText] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const loading = status === "loading";

  // Keep the newest message in view.
  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [messages, status]);

  async function request(text: string) {
    setStatus("loading");
    setFailedText(null);
    const result = await sendMessage(text, threadId);

    switch (result.kind) {
      case "ok":
        setThreadId(result.threadId);
        setMessages((prev) => [...prev, { id: nextId++, role: "assistant", text: result.response }]);
        setStatus("idle");
        break;
      case "blocked":
        setInput(text); // give the text back so it can be rephrased
        setStatus("blocked");
        break;
      case "rate_limited":
        setStatus("rate_limited");
        break;
      case "error":
        setFailedText(text);
        setStatus("error");
        break;
    }
  }

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setMessages((prev) => [...prev, { id: nextId++, role: "user", text: trimmed }]);
    setInput("");
    void request(trimmed);
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    submit(input);
  }

  // Enter sends; Shift+Enter inserts a new line.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit(input);
    }
  }

  function newConversation() {
    setMessages([]);
    setThreadId(null);
    setInput("");
    setStatus("idle");
    setFailedText(null);
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Weather Agent</h1>
        <button
          type="button"
          className="button-secondary"
          onClick={newConversation}
          disabled={loading}
        >
          New conversation
        </button>
      </header>

      <main className="log" ref={logRef} role="log" aria-label="Conversation">
        {messages.length === 0 && (
          <div className="welcome">
            <p>Ask about live weather or historical air quality.</p>
            <div className="chips">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="chip"
                  onClick={() => submit(example)}
                  disabled={loading}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`message message-${message.role}`}>
            <span className="visually-hidden">
              {message.role === "user" ? "You said:" : "Agent replied:"}
            </span>
            {message.text}
          </div>
        ))}

        {loading && (
          <div className="message message-assistant thinking" role="status">
            Agent is thinking…
          </div>
        )}

        {(status === "blocked" || status === "rate_limited" || status === "error") && (
          <div className={`notice notice-${status}`} role="alert">
            <span>{NOTICES[status]}</span>
            {status === "error" && failedText !== null && (
              <button type="button" className="button-secondary" onClick={() => request(failedText)}>
                Retry
              </button>
            )}
          </div>
        )}
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <label htmlFor="message" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="message"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          maxLength={MAX_MESSAGE_LENGTH}
          rows={2}
          placeholder="Ask about the weather or air quality…"
          aria-describedby="char-count"
        />
        <div className="composer-footer">
          <span id="char-count" className="char-count">
            {input.length}/{MAX_MESSAGE_LENGTH}
          </span>
          <button type="submit" className="button-primary" disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
