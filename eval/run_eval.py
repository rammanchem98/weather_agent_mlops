import json
import uuid
import time
from google.genai import types
from src.agent.graph import compiled_agent
from eval.judge import judge_answer

def load_golden_queries(path="eval/golden_queries.jsonl"):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_called_tools(result: dict) -> list[str]:
    called = []
    for msg in result["messages"]:
        for part in (msg.parts or []):
            if part.function_call:
                called.append(part.function_call.name)
    return called

# Eval manages running the agent on a set of golden queries, checking which tools were called,
#  and optionally judging the final answer's grounding against the tool results. (MANUAL HARNESSING))
def run_case(case: dict) -> dict:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    user_message = types.Content(role="user", parts=[types.Part.from_text(text=case["query"])])
    result = compiled_agent.invoke({"messages": [user_message]}, config=config)

    actual_tools = get_called_tools(result)
    expected_tools = set(case["expected_tools"])
    tools_passed = set(actual_tools) == expected_tools

    judge_result = None
    if case.get("check_grounding", False):
        tool_results, final_answer = get_tool_results_and_answer(result)
        judge_result = judge_answer(case["query"], tool_results, final_answer)

    overall_passed = tools_passed and (judge_result is None or judge_result["grounded"])

    return {
        "id": case["id"], "query": case["query"],
        "expected": sorted(expected_tools), "actual": sorted(set(actual_tools)),
        "tools_passed": tools_passed, "judge_result": judge_result,
        "passed": overall_passed,
    }

# To get the tool results and final answer from the agent's result, used in jugge.py
def get_tool_results_and_answer(result: dict) -> tuple[list[str], str]:
    tool_results = []
    for msg in result["messages"]:
        for part in (msg.parts or []):
            if part.function_response:
                # function_response.response is usually a dict; adjust based on your actual structure
                tool_results.append(str(part.function_response.response))

    final_answer = ""
    last = result["messages"][-1]
    for part in (last.parts or []):
        if part.text:
            final_answer = part.text
    return tool_results, final_answer

def main():
    cases = load_golden_queries()
    results = []
    for c in cases:
        results.append(run_case(c))
        time.sleep(15)
    print(f"\n{'='*60}\nEVAL RESULTS\n{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if not r["passed"]:
            if not r["tools_passed"]:
                print(f"    expected: {r['expected']}")
                print(f"    actual:   {r['actual']}")
            if r["judge_result"] and not r["judge_result"]["grounded"]:
                print(f"    grounding issue: {r['judge_result']['reason']}")
                
    passed_count = sum(r["passed"] for r in results)
    print(f"\n{passed_count}/{len(results)} passed\n")

if __name__ == "__main__":
    main()