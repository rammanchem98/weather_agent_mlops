import json
import uuid
from google.genai import types
from src.agent.graph import compiled_agent

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

def run_case(case: dict) -> dict:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    user_message = types.Content(role="user", parts=[types.Part.from_text(text=case["query"])])
    result = compiled_agent.invoke({"messages": [user_message]}, config=config)

    actual_tools = get_called_tools(result)
    expected_tools = set(case["expected_tools"])
    passed = set(actual_tools) == expected_tools

    return {
        "id": case["id"], "query": case["query"],
        "expected": sorted(expected_tools), "actual": sorted(set(actual_tools)),
        "passed": passed,
    }

def main():
    cases = load_golden_queries()
    results = [run_case(c) for c in cases]
    print(f"\n{'='*60}\nEVAL RESULTS\n{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if not r["passed"]:
            print(f"    query:    {r['query']}")
            print(f"    expected: {r['expected']}")
            print(f"    actual:   {r['actual']}")
    passed_count = sum(r["passed"] for r in results)
    print(f"\n{passed_count}/{len(results)} passed\n")

if __name__ == "__main__":
    main()