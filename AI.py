import ollama
import json
from datetime import datetime
import sys
import time

# Load past chats
try:
    with open("memory.json", "r") as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = []


def chat(input_text):
    global memory  # Allow modifying the global memory

    # Add user message to memory
    memory.append({"role": "user", "content": input_text, "time": str(datetime.now())})

    # Generate response
    response = ollama.chat(
        model='phi3',
        messages=[{"role": "user", "content": input_text}],  # Comma added here
        options={
            "num_predict": 64,  # Shorter responses
            "temperature": 0.7,  # Less randomness = faster
            "seed": 123,  # Consistent outputs
            "num_ctx": 1024  # Smaller context window
        }
    )

    # Extract the actual response content
    ai_response = response['message']['content']

    # Add AI reply to memory
    memory.append({"role": "assistant", "content": ai_response, "time": str(datetime.now())})

    # Memory management
    if len(memory) > 20:  # Keep last 20 exchanges
        memory = memory[-20:]

    if len(memory) > 10:
        summary_prompt = f"Summarize this concisely:\n{str(memory[:-5])}"
        summary = ollama.generate(model='phi3', prompt=summary_prompt)
        memory = [{"role": "system", "content": "Previous chats summarized: " + summary}] + memory[-5:]

    # Save to disk
    with open("memory.json", "w") as f:
        json.dump(memory, f, indent=2)

    return ai_response


def print_typing():
    for char in "AI is thinking... ":
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)
    print()


# CLI interface
print("Your AI: Hi! Ask or tell me anything. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    print_typing()

    # Choose EITHER streaming OR regular chat:

    # Option 1: Regular chat (uses the chat() function with memory)
    response = chat(user_input)
    print("AI:", response)
