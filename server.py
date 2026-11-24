from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from google.genai import types
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import json
import os
import uuid
import logging
import time
from datetime import datetime

from src.agent.graph import compiled_agent
from src.agent.trace import TRACE_CONFIG_KEY, TraceRecorder, build_trace
from src.api.guardrails import check_input, check_output
from src.api.rate_limit import chat_rate_limiter, client_ip
from src.api.static import mount_frontend
from src.common.config import ROOT_DIR
from src.common.redact import redact_secrets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUEST_COUNT = Counter("agent_requests_total", "Total chat requests", ["status"])
REQUEST_LATENCY = Histogram("agent_request_duration_seconds", "Request latency")

app = FastAPI(title="Weather Agent API")

AGENT_ERROR_MESSAGE = "The agent couldn't complete this request. Please try again."


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    trace: dict | None = None  # optional; existing fields are unchanged


def sanitize_for_client(text: str) -> str:
    """Every string in the trace goes through the same filters as the answer."""
    return check_output(redact_secrets(text))


def guardrail_summary(verdict: dict) -> dict:
    return {layer: verdict[layer] for layer in ("regex", "classifier") if layer in verdict}


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    start_time = time.time()  # renamed from "start" to avoid any confusion with the tracemalloc import

    # Checked before the guardrails so rejected spam doesn't cost a classifier call
    if not chat_rate_limiter.allow(client_ip(http_request)):
        REQUEST_COUNT.labels(status="rate_limited").inc()
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")

    recorder = TraceRecorder()  # one per request; never shared between requests
    verdict: dict = {}
    is_safe, rejection_reason = await check_input(request.message, verdict)
    recorder.guardrail = guardrail_summary(verdict)
    recorder.count_gemini_call(verdict.get("classifier_calls", 0))
    if not is_safe:
        REQUEST_COUNT.labels(status="rejected").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        raise HTTPException(status_code=400, detail=rejection_reason)

    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, TRACE_CONFIG_KEY: recorder}}
    user_message = types.Content(role="user", parts=[types.Part.from_text(text=request.message)])

    try:
        result = await compiled_agent.ainvoke({"messages": [user_message]}, config=config)
    except Exception:
        # Full exception (type, message, traceback) goes to the server log only;
        # the client gets a generic message so internals never leak.
        logging.exception("Agent invocation failed")
        REQUEST_COUNT.labels(status="error").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=AGENT_ERROR_MESSAGE)

    last = result["messages"][-1]
    final_text = "".join(part.text for part in (last.parts or []) if part.text and not part.thought)
    safe_text = check_output(final_text)

    REQUEST_COUNT.labels(status="success").inc()
    elapsed = time.time() - start_time
    REQUEST_LATENCY.observe(elapsed)
    trace = build_trace(recorder, elapsed, sanitize_for_client)
    return ChatResponse(response=safe_text, thread_id=thread_id, trace=trace)


@app.get("/health")
async def health():
    return {"status": "ok"}


EVAL_RESULTS_PATH = os.path.join(ROOT_DIR, "eval", "results.json")


def _load_eval_summary(path: str) -> dict | None:
    """Only run_at (ISO date string), passed and total (ints); None if the file
    is missing or anything about it is malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    run_at, passed, total = data.get("run_at"), data.get("passed"), data.get("total")
    if not isinstance(run_at, str) or not 0 < len(run_at) <= 40:
        return None
    try:
        datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    ints_ok = all(isinstance(v, int) and not isinstance(v, bool) for v in (passed, total))
    if not ints_ok or total <= 0 or not 0 <= passed <= total:
        return None
    return {"run_at": run_at, "passed": passed, "total": total}


@app.get("/eval/summary")
async def eval_summary():
    summary = _load_eval_summary(EVAL_RESULTS_PATH)
    if summary is None:
        raise HTTPException(status_code=404, detail="No eval results.")  # never a path or file contents
    return summary


# Must stay last: the front end is mounted at "/", which would shadow any route added after it
mount_frontend(app)