import gradio as gr
import json
import os
import pyttsx3
import ollama

print("🚀 Adarsh AI Clone Starting...")

# YOUR PERSONALITY (Simple list - No ChromaDB needed)
PERSONALITY = [
    "I'm Adarsh Singh, CSE student at Manipal University Jaipur, Reg No: 2427030325.",
    "I love Python programming and building AI projects.",
    "Debug ML models: data first → hyperparameters → architecture.",
    "PBL projects: working prototypes > complex theory.",
    "Communication: direct, technical, practical solutions.",
    "Current project: AI Human Clone using Ollama.",
    "Guide: Mr. Virendra Mehgal."
]

# VOICE SETUP
tts = pyttsx3.init()
tts.setProperty('rate', 170)

# CHAT HISTORY
HISTORY_FILE = "chat_history.json"
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def chat_with_clone(message, history):
    # Load past conversations
    past_history = load_history()
    
    # Build context from personality + recent chats
    context = "\n".join(PERSONALITY)
    chat_context = "\n".join([f"U: {h[0]}\nA: {h[1]}" for h in past_history[-5:]])
    
    prompt = f"""You are Adarsh Singh (CSE Manipal student, Reg: 2427030325). 
Respond EXACTLY like him - direct, technical, practical.

PERSONALITY:
{context}

RECENT CHATS:
{chat_context}

USER: {message}
Adarsh:"""
    
    # Ollama generates response
    response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
    reply = response['message']['content']
    
    # Save to history
    past_history.append([message, reply])
    with open(HISTORY_FILE, 'w') as f:
        json.dump(past_history, f)
    
    # SPEAK!
    tts.say(reply)
    tts.runAndWait()
    
    # Update Gradio chat
    history.append([message, reply])
    return history, ""

# LAUNCH UI
demo = gr.ChatInterface(
    chat_with_clone,
    title="🧑‍💻 Adarsh AI Clone v2.0",
    description="Manipal University Jaipur PBL | Ollama + Voice | Python 3.14 Compatible",
    examples=["Tell me about yourself", "How do you debug ML?", "What is your registration number?"]
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
