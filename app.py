import gradio as gr
import json
import os
import ollama
import threading
import time

print("🚀 Adarsh AI Clone v9.2 - BUGFIXED")

# ===================== PERSONALITY =====================
PERSONALITY = """
You are roleplaying as Adarsh Singh. Stay in character at ALL times.

Identity:
- Name: Adarsh Singh
- Course: CSE, 4th Semester
- University: Manipal University Jaipur
- Registration Number: 2427030325
- Guide: Mr. Virendra Mehgal

Communication style:
- Direct and slightly jolly
- Use SIMPLE English ONLY — no Hindi, no Hinglish
- Short sentences, no over-explaining

Skills: Python, AI/ML basics, debugging models

Hobbies (STRICT - only these):
- Sports: judo, volleyball, football
- Non-sport: listening to music
- NEVER say "outdoor sports", NEVER mention cricket or tennis

Project: AI Digital Twin using Ollama, JSON memory, gTTS voice, Gradio UI

Behavior:
- Always answer in English only
- Keep answers under 3 sentences unless the topic needs more
- Never break character
- Never say you're an AI language model — you are Adarsh Singh
"""

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
            json.dump(data, f, indent=2)
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

    # Load last 3 exchanges for context
    past = load_history()[-3:]
    context = "\n".join([f"User: {h[0]}\nAdarsh: {h[1]}" for h in past if h[1]])

    prompt = f"""{PERSONALITY}

--- RECENT CONVERSATION ---
{context if context else "(no prior context)"}

--- CURRENT MESSAGE ---
User: {message}
Adarsh:"""

    try:
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[
                {
                    'role': 'system',
                    'content': PERSONALITY
                },
                {
                    'role': 'user',
                    'content': f"Recent context:\n{context}\n\nUser says: {message}\n\nRespond as Adarsh Singh in English only, briefly:"
                }
            ]
        )

        reply = ""
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            reply = response.message.content
        elif isinstance(response, dict):
            reply = response.get('message', {}).get('content', '')

        reply = reply.strip()

        # Remove "Adarsh:" prefix if model repeats it
        if reply.lower().startswith("adarsh:"):
            reply = reply[7:].strip()

        # Truncate at 300 chars but don't cut mid-sentence
        if len(reply) > 300:
            cutoff = reply[:300].rfind('.')
            reply = reply[:cutoff + 1] if cutoff > 100 else reply[:300]

        if not reply:
            reply = "Hey! Ask me anything — about my project, hobbies, or CSE stuff."

        speak_async(reply)

    except Exception as e:
        print("🔥 OLLAMA ERROR:", str(e))
        reply = f"❌ Ollama error: {str(e)}\n\nMake sure Ollama is running: `ollama serve` and model is pulled: `ollama pull llama3.2:1b`"

    # Save to memory
    all_history = load_history()
    all_history.append([message, reply])
    save_history(all_history)

    return reply

# ===================== CUSTOM CSS =====================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --accent: #22d3ee;
    --bg-dark: #0f0f13;
    --bg-card: #16161d;
    --bg-input: #1e1e2a;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --border: #2d2d3d;
    --user-bubble: #312e81;
    --bot-bubble: #1a1a27;
    --success: #10b981;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.gradio-container h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1, #22d3ee) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.5px !important;
}

.gradio-container .prose p {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}

/* Chat container */
.chatbot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 40px rgba(99, 102, 241, 0.08) !important;
}

/* Messages */
.message-wrap {
    padding: 8px 16px !important;
}

.message.user {
    background: var(--user-bubble) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #c7d2fe !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

.message.bot {
    background: var(--bot-bubble) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px 14px 14px 4px !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

/* Input area */
.input-row {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
    padding: 4px !important;
}

textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    border: none !important;
    resize: none !important;
}

textarea:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* Send button */
button.primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

/* Example buttons */
.examples-holder button {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
}

.examples-holder button:hover {
    border-color: var(--primary) !important;
    color: var(--text) !important;
}

/* Status/footer */
.generating {
    color: var(--accent) !important;
    font-size: 0.75rem !important;
}

/* Avatar */
.icon-button-wrapper {
    display: none !important;
}
"""

# ===================== UI =====================
with gr.Blocks(title="Adarsh AI Clone") as demo:
    gr.Markdown("""
# 🧠 Adarsh AI Clone
**AI Digital Twin** · Manipal University Jaipur · PBL-2 · Local LLM via Ollama
""")

    chatbot = gr.ChatInterface(
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
            "Explain your project architecture",
            "How do you debug ML models?",
            "What's your registration number?",
        ],
    )

    gr.Markdown("""
<div style="text-align:center; color:#475569; font-size:0.75rem; margin-top:12px;">
Powered by Ollama · llama3.2:1b · gTTS Voice · JSON Memory
</div>
""")

# ===================== RUN =====================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)
