// Typed client for the FastAPI backend (server.py).
//
// The UI never sees HTTP status codes or the server's `detail` text: every
// outcome is mapped to a small ChatResult union here. That keeps raw error
// details (guardrail reasons, Python exception messages) out of the page.

// Mirrors MAX_MESSAGE_LENGTH in src/api/guardrails.py.
export const MAX_MESSAGE_LENGTH = 2000;

export type ChatResult =
  | { kind: "ok"; response: string; threadId: string }
  | { kind: "blocked" } // 400: guardrail or validation rejection
  | { kind: "rate_limited" } // 429: too many messages from this client
  | { kind: "error" }; // 500, any other status, bad JSON, or network failure

interface ChatResponseBody {
  response: string;
  thread_id: string;
}

function isChatResponseBody(value: unknown): value is ChatResponseBody {
  const body = value as ChatResponseBody;
  return typeof body?.response === "string" && typeof body?.thread_id === "string";
}

export async function sendMessage(
  message: string,
  threadId: string | null,
): Promise<ChatResult> {
  let res: Response;
  try {
    res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });
  } catch {
    return { kind: "error" }; // network failure: server down, offline, etc.
  }

  if (res.status === 400) return { kind: "blocked" };
  if (res.status === 429) return { kind: "rate_limited" };
  if (!res.ok) return { kind: "error" };

  try {
    const body: unknown = await res.json();
    if (!isChatResponseBody(body)) return { kind: "error" };
    return { kind: "ok", response: body.response, threadId: body.thread_id };
  } catch {
    return { kind: "error" };
  }
}
