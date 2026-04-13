import gradio as gr
import json
import os
import ollama
import threading
import time

print("🚀 Adarsh AI Clone v9.3 - STABLE")

# ===================== PERSONALITY =====================
SYSTEM_PROMPT = """You are Adarsh Singh, a CSE student at Manipal University Jaipur.

Facts about you:
- Name: Adarsh Singh
- Registration Number: 2427030325
- Course: CSE, 4th Semester, 2nd Year
- University: Manipal University Jaipur
- Guide: Mr. Virendra Mehgal
- Project: AI Digital Twin using Ollama, Gradio, gTTS, JSON memory

Your hobbies (ONLY these, never say anything else):
- Judo, volleyball, football
- Listening to music

Your skills: Python, AI/ML basics, debugging models

How you talk:
- Always in English only
- Short and direct answers
- Slightly friendly tone
- Never over-explain

STRICT RULES:
- Answer ONLY what is asked right now
- Do NOT mention previous conversations
- Do NOT repeat old answers
- Do NOT say "as I said before" or anything like that
- Do NOT mention cricket, tennis, or outdoor sports"""

# ===================== MEMORY =====================
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(data):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

# ===================== VOICE =====================
def speak_async(text):
    def _speak():
        try:
            from gtts import gTTS
            from playsound import playsound

            filename = f"voice_{int(time.time()*1000)}.mp3"
            tts = gTTS(text=text[:200], lang='en', tld='com.au')
            tts.save(filename)
            playsound(filename)
            try:
                os.remove(filename)
            except:
                pass
        except Exception as e:
            print("Voice Error:", e)

    threading.Thread(target=_speak, daemon=True).start()

# ===================== MAIN CHAT =====================
def chat_with_clone(message, history):
    time.sleep(0.2)

    try:
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': message}
            ]
        )

        # Parse response
        if hasattr(response, 'message'):
            reply = response.message.content
        elif isinstance(response, dict):
            reply = response.get('message', {}).get('content', '')
        else:
            reply = str(response)

        reply = reply.strip()

        # Remove echoed "Adarsh:" prefix if model adds it
        if reply.lower().startswith("adarsh:"):
            reply = reply[7:].strip()

        # Truncate at sentence boundary within 280 chars
        if len(reply) > 280:
            cutoff = reply[:280].rfind('.')
            reply = reply[:cutoff + 1] if cutoff > 80 else reply[:280]

        if not reply:
            reply = "Hey, ask me anything!"

        speak_async(reply)

    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error: {str(e)}"

    # Save to memory
    all_history = load_history()
    all_history.append([message, reply])
    save_history(all_history)

    return reply

# ===================== CUSTOM CSS =====================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

body, .gradio-container {
    background: #0f0f13 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #e2e8f0 !important;
}

/* Header */
.gradio-container h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1, #22d3ee) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.gradio-container .prose p {
    color: #64748b !important;
    font-size: 0.85rem !important;
}

/* Chat container */
.chatbot {
    background: #161622 !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 40px rgba(99,102,241,0.08) !important;
}

/* User message bubble */
.message.user {
    background: #312e81 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #c7d2fe !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

/* Bot message bubble */
.message.bot {
    background: #1a1a27 !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 14px 14px 14px 4px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

/* Input area */
textarea {
    background: #1e1e2a !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 10px !important;
}

textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    outline: none !important;
}

/* Send button */
button.primary {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}

/* Example buttons */
.examples-holder button {
    background: #1e1e2a !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
}

.examples-holder button:hover {
    border-color: #6366f1 !important;
    color: #e2e8f0 !important;
}
"""

# ===================== UI =====================
with gr.Blocks(title="Adarsh AI Clone") as demo:
    gr.Markdown("""
# 🧠 Adarsh AI Clone
**AI Digital Twin** · Manipal University Jaipur · PBL-2 · Local LLM via Ollama
""")

    gr.ChatInterface(
        fn=chat_with_clone,
        chatbot=gr.Chatbot(
            height=460,
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/initials/svg?seed=AS&backgroundColor=6366f1"),
        ),
        textbox=gr.Textbox(
            placeholder="Ask me anything — hobbies, project, CSE stuff...",
            show_label=False,
            lines=1,
        ),
        examples=[
            "Tell me about yourself",
            "What are your hobbies?",
            "What is your registration number?",
            "How do you debug ML models?",
        ],
    )

    gr.Markdown("""
<div style="text-align:center; color:#475569; font-size:0.75rem; margin-top:8px;">
Powered by Ollama · llama3.2:1b · gTTS Voice · JSON Memory
</div>
""")

# ===================== RUN =====================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)
