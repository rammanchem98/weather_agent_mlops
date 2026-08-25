import uuid
from google.genai import types
from src.agent.graph import compiled_agent

def ask(question: str):
          thread_id = str(uuid.uuid4())
          config = {"configurable": {"thread_id": thread_id}}
          user_message = types.Content(role="user", parts=[types.Part.from_text(text=question)])
          result = compiled_agent.invoke({"messages": [user_message]}, config=config)

          print(f"\n{'='*60}\nQ: {question}")
          for msg in result["messages"]:
              for part in (msg.parts or []):
                  if part.function_call:
                      print(f"  [tool call] {part.function_call.name}({dict(part.function_call.args)})")
          last = result["messages"][-1]
          for part in last.parts:
            if part.text:
                print(f"A: {part.text}")

if __name__ == "__main__":
    ask("What is the current weather in Tokyo?")
    ask("What was the PM2.5 and temperature for Beijing in your historical database?")
    ask("Compare live weather in London to the historical air quality record you have for London.")

# run this in powershell first  $env:PYTHONIOENCODING="utf-8" 