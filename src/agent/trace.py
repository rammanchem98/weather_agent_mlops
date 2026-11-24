"""Per-request record of what the agent did, returned to the UI as `trace`.

One TraceRecorder is created per /chat request and handed to the graph through
LangGraph's per-invocation config (key TRACE_CONFIG_KEY), so concurrent
requests never share one. Nodes record into it only if it is present; eval,
main.py and tests that don't pass one are unaffected.

build_trace() is the only way data leaves: every string goes through the
caller's sanitizer (redaction + check_output) and tool results are truncated."""
from dataclasses import dataclass, field
from typing import Any, Callable

# Leading "__": LangGraph keeps double-underscore configurable keys out of
# checkpoint metadata, so the recorder object is never serialized by MemorySaver.
TRACE_CONFIG_KEY = "__trace_recorder"
RESULT_PREVIEW_CHARS = 300
PLAN_MAX_CHARS = 2000  # sanity cap; the UI shows ~300 chars with "Show more"


@dataclass
class TraceRecorder:
    guardrail: dict = field(default_factory=dict)
    gemini_calls: int = 0          # attempted generateContent calls, failed ones included
    tool_calls: list = field(default_factory=list)
    rounds: int = 0                # tool rounds that actually ran
    capped: bool = False           # the tool-loop cap ended the turn
    thoughts: list = field(default_factory=list)

    def count_gemini_call(self, n: int = 1) -> None:
        self.gemini_calls += n

    def add_tool_round(self, calls: list[dict]) -> None:
        self.rounds += 1
        self.tool_calls.extend(calls)

    def add_thought(self, text: str) -> None:
        if text and text.strip():
            self.thoughts.append(text.strip())


def get_recorder(config) -> "TraceRecorder | None":
    if not config:
        return None
    recorder = (config.get("configurable") or {}).get(TRACE_CONFIG_KEY)
    return recorder if isinstance(recorder, TraceRecorder) else None


def _sanitize_value(value: Any, sanitize: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, dict):
        return {sanitize(str(k)): _sanitize_value(v, sanitize) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v, sanitize) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return sanitize(str(value))


def build_trace(recorder: TraceRecorder, total_latency_s: float, sanitize: Callable[[str], str]) -> dict:
    trace = {
        "guardrail": _sanitize_value(dict(recorder.guardrail), sanitize),
        "tool_calls": [
            {
                "name": sanitize(str(call.get("name", ""))),
                "args": _sanitize_value(call.get("args") or {}, sanitize),
                # sanitize before truncating, so a cut can't split a secret past the redactor
                "result_preview": sanitize(str(call.get("result", "")))[:RESULT_PREVIEW_CHARS],
                "duration_s": round(float(call.get("duration_s") or 0.0), 3),
            }
            for call in recorder.tool_calls
        ],
        "rounds": recorder.rounds,
        "capped": recorder.capped,
        "gemini_calls": recorder.gemini_calls,
        "total_latency_s": round(total_latency_s, 3),
    }
    if recorder.thoughts:
        plan = sanitize("\n\n".join(recorder.thoughts))  # sanitize first, then cut
        if len(plan) > PLAN_MAX_CHARS:
            plan = plan[:PLAN_MAX_CHARS - 1].rstrip() + "…"
        trace["plan"] = plan
    return trace
