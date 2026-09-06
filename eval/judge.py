# eval/judge.py
import json
from google import genai

client = genai.Client()
JUDGE_MODEL = "gemini-2.5-flash"

JUDGE_PROMPT = """You are grading whether an AI agent's final answer is factually
grounded in the tool results it was given. You are NOT grading writing style,
tone, or completeness — only factual accuracy against the provided evidence.

User question: {query}

Tool results the agent had access to:
{tool_results}

Agent's final answer:
{final_answer}

Does the final answer contain any claim that CONTRADICTS or is NOT SUPPORTED
by the tool results above? Minor rounding or rephrasing is fine — flag only
genuine factual mismatches (wrong numbers, wrong city, invented data).

Respond with ONLY valid JSON, no other text:
{{"grounded": true or false, "reason": "one sentence explaining your verdict"}}
"""

def judge_answer(query: str, tool_results: list[str], final_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        query=query,
        tool_results="\n".join(tool_results) if tool_results else "(no tools were called)",
        final_answer=final_answer,
    )
    response = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)

    try:
        return json.loads(response.text.strip())
    except json.JSONDecodeError:
        # judge itself misbehaved — don't silently pass, surface it as a failure to investigate
        return {"grounded": False, "reason": f"Judge returned unparseable output: {response.text[:200]}"}