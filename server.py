# server.py
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import types

from src.agent.graph import compiled_agent
from src.api.guardrails import check_input, check_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Weather Agent API")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # optional — lets a client continue a conversation


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    is_safe, rejection_reason = await check_input(request.message)
    if not is_safe:
        raise HTTPException(status_code=400, detail=rejection_reason)

    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    user_message = types.Content(role="user", parts=[types.Part.from_text(text=request.message)])

    try:
        result = await compiled_agent.ainvoke({"messages": [user_message]}, config=config)
    except Exception as e:
        logging.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    last = result["messages"][-1]
    final_text = "".join(part.text for part in (last.parts or []) if part.text)
    safe_text = check_output(final_text)

    return ChatResponse(response=safe_text, thread_id=thread_id)


@app.get("/health")
async def health():
    return {"status": "ok"}