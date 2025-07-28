import ollama
import json
from datetime import datetime
import sys
import time

# Initialize memory
try:
    with open("memory.json", "r") as f:
        memory = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    memory = [{"role": "system", "content": "You are a helpful AI assistant."}]


def chat(user_input, stream=False):
    global memory

    # Add user message
    memory.append({"role": "user", "content": user_input})

    try:
        if stream:
            # Streaming response
            print("AI: ", end="", flush=True)
            stream = ollama.chat(
                model='phi3',
                messages=[m for m in memory if m["role"] != "time"],
                stream=True,
                options={
                    "num_predict": 128,
                    "temperature": 0.7,
                    "num_ctx": 2048
                }
            )

            ai_response = ""
            for chunk in stream:
                print(chunk['message']['content'], end="", flush=True)
                ai_response += chunk['message']['content']
            print()

        else:
            # Regular response
            response = ollama.chat(
                model='phi3',
                messages=[m for m in memory if m["role"] != "time"],
                options={
                    "num_predict": 128,
                    "temperature": 0.7,
                    "num_ctx": 2048
                }
            )
            ai_response = response['message']['content']
            print("AI:", ai_response)

        # Add AI response
        memory.append({"role": "assistant", "content": ai_response})

        # Maintain memory size
        if len(memory) > 20:
            # Try to summarize if too long
            try:
                summary = ollama.generate(
                    model='phi3',
                    prompt=f"Summarize this conversation in 1-2 sentences:\n{str(memory[:-10])}"
                )['response']
                memory = [memory[0],
                          {"role": "system", "content": f"Previous conversation summary: {summary}"}] + memory[-10:]
            except:
                memory = memory[-15:]  # Fallback truncation

        # Save to disk
        with open("memory.json", "w") as f:
            json.dump(memory, f, indent=2)

        return ai_response

    except Exception as e:
        print(f"\nError: {str(e)}")
        return "Sorry, I encountered an error."


def print_typing():
    for char in "AI is thinking... ":
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)
    print()


# Main loop
print("Your AI: Hi! Ask or tell me anything. Type 'quit' to exit.")
print("Tip: Start your message with 'stream' for word-by-word response")
while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break

    stream_mode = user_input.lower().startswith('stream')
    if stream_mode:
        user_input = user_input[6:].strip()  # Remove 'stream' prefix

    print_typing()
    chat(user_input, stream=stream_mode)