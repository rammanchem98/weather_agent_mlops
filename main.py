import uuid
from google.genai import types
from src.agent.graph import compiled_agent

def main():
    print("=" * 60)
    print("🌍 AIR QUALITY & WEATHER AGENT INITIALIZED")
    print("Type your questions below ('q' to quit).")
    print("=" * 60)

    # Unique thread ID preserves state across user turns
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Session terminated.")
                break

            user_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )

            # Invoke state graph
            result = compiled_agent.invoke(
                {"messages": [user_message]},
                config=config
            )

            # Retrieve final assistant output
            last_message = result["messages"][-1]
            if hasattr(last_message, "parts"):
                for part in last_message.parts:
                    if part.text:
                        print(f"\n🤖 Assistant:\n{part.text}")

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
