// Typed client for the FastAPI backend (server.py).
//
// The UI never sees HTTP status codes or the server's `detail` text: every
// outcome is mapped to a small ChatResult union here. That keeps raw error
// details (guardrail reasons, Python exception messages) out of the page.

// Mirrors MAX_MESSAGE_LENGTH in src/api/guardrails.py.
export const MAX_MESSAGE_LENGTH = 2000;

export type ChatResult =
  | { kind: "ok"; response: string; threadId: string; trace?: Trace }
  | { kind: "blocked" } // 400: guardrail or validation rejection
  | { kind: "rate_limited" } // 429: too many messages from this client
  | { kind: "error" }; // 500, any other status, bad JSON, or network failure

export interface ToolCallTrace {
  name: string;
  args: Record<string, unknown>;
  resultPreview: string;
  durationS?: number;
}

// What the agent did for one reply. Every field is optional: an older server,
// or a malformed trace, just means less (or nothing) is shown.
export interface Trace {
  guardrail: Record<string, string>;
  plan?: string;
  toolCalls: ToolCallTrace[];
  rounds?: number;
  capped?: boolean;
  geminiCalls?: number;
  totalLatencyS?: number;
}

export interface EvalSummary {
  runAt: string;
  passed: number;
  total: number;
}

interface ChatResponseBody {
  response: string;
  thread_id: string;
  trace?: unknown;
}

function isChatResponseBody(value: unknown): value is ChatResponseBody {
  const body = value as ChatResponseBody;
  return typeof body?.response === "string" && typeof body?.thread_id === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const num = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : undefined);

// Keeps only well-typed fields; returns undefined if there is nothing usable.
export function parseTrace(value: unknown): Trace | undefined {
  if (!isRecord(value)) return undefined;

  const guardrail: Record<string, string> = {};
  if (isRecord(value.guardrail)) {
    for (const [layer, status] of Object.entries(value.guardrail)) {
      if (typeof status === "string") guardrail[layer] = status;
    }
  }

  const toolCalls: ToolCallTrace[] = Array.isArray(value.tool_calls)
    ? value.tool_calls.filter(isRecord).flatMap((call) =>
        typeof call.name === "string"
          ? [{
              name: call.name,
              args: isRecord(call.args) ? call.args : {},
              resultPreview: typeof call.result_preview === "string" ? call.result_preview : "",
              durationS: num(call.duration_s),
            }]
          : [],
      )
    : [];

  return {
    guardrail,
    plan: typeof value.plan === "string" && value.plan.trim() ? value.plan : undefined,
    toolCalls,
    rounds: num(value.rounds),
    capped: typeof value.capped === "boolean" ? value.capped : undefined,
    geminiCalls: num(value.gemini_calls),
    totalLatencyS: num(value.total_latency_s),
  };
}

// Null when there are no (valid) eval results; the footer is then hidden.
export async function fetchEvalSummary(): Promise<EvalSummary | null> {
  try {
    const res = await fetch("/eval/summary");
    if (!res.ok) return null;
    const body: unknown = await res.json();
    if (!isRecord(body)) return null;
    const { run_at: runAt, passed, total } = body;
    if (typeof runAt !== "string" || !Number.isInteger(passed) || !Number.isInteger(total)) return null;
    return { runAt, passed: passed as number, total: total as number };
  } catch {
    return null;
  }
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
    return { kind: "ok", response: body.response, threadId: body.thread_id, trace: parseTrace(body.trace) };
  } catch {
    return { kind: "error" };
  }
}
