import argparse
import sys
import requests

def main():
    parser = argparse.ArgumentParser(description="NexoraLM Interactive Chat CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="API server base URL")
    args = parser.parse_args()

    print(f"Connecting to NexoraLM API at {args.url}...")
    try:
        health = requests.get(f"{args.url}/health").json()
        print(f"Server Status: {health.get('status')}")
    except Exception as e:
        print(f"Failed to connect to API server: {e}")
        print("Please start server with: python scripts/serve.py")
        sys.exit(1)

    messages = [
        {"role": "system", "content": "You are NexoraLM, a compact and helpful AI assistant."}
    ]

    print("NexoraLM Chat CLI ready. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat session.")
                break

            messages.append({"role": "user", "content": user_input})

            res = requests.post(
                f"{args.url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                }
            )

            if res.status_code == 200:
                reply = res.json()["choices"][0]["message"]["content"]
                print(f"NexoraLM: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
            else:
                print(f"API Error {res.status_code}: {res.text}\n")

        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break

if __name__ == "__main__":
    main()
