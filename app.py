"""
app.py — Adarsh AI Clone v9.4
Main entry point. UI + chat logic only.
All memory, voice, and identity logic lives in backend.py.
"""

import gradio as gr
import ollama

# Import everything from backend — no more duplicated functions
from backend import (
    load_history,
    save_history,
    speak_async,
    clean_reply,
    load_identity,
    build_system_prompt,
    build_messages,
)

print("🚀 Adarsh AI Clone v9.4 - Memory Fixed")

# ===================== IDENTITY =====================
# Try loading from identity.json first; fall back to hardcoded if missing
_identity = load_identity("identity.json")
if _identity:
    SYSTEM_PROMPT = build_system_prompt(_identity)
    print("✅ Identity loaded from identity.json")
else:
    # Hardcoded fallback — works even without identity.json
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
- Do NOT say 'as I said before' or anything like that
- Do NOT mention cricket, tennis, or sports not in your hobbies list"""
    print("⚠️  identity.json not found — using hardcoded fallback")


# ===================== MAIN CHAT =====================
def chat_with_clone(message, history):
    """
    Core chat function called by Gradio on every message.

    THE KEY FIX (history bleeding bug):
    Previously, history was either not used at all, or injected as raw text
    which caused previous refusals to bleed into new unrelated questions.

    Now we use build_messages() which passes history in Ollama's native
    role-based format (user/assistant pairs). The model correctly treats
    each past turn as separate context — a drug refusal on day 1 will NOT
    affect an Iran-Israel question on day 2.
    """
    try:
        # Build proper multi-turn messages list (system + past turns + current)
        messages = build_messages(SYSTEM_PROMPT, message, n=5)

        response = ollama.chat(
            model="llama3.2:1b",
            messages=messages
        )

        # Parse response — handles both object-style and dict-style SDK responses
        if hasattr(response, "message"):
            reply = response.message.content
        elif isinstance(response, dict):
            reply = response.get("message", {}).get("content", "")
        else:
            reply = str(response)

        # Clean the reply (strip echoed prefix, markdown, Hindi chars, truncate)
        reply = clean_reply(reply, max_chars=300)

        # Speak the reply in background (non-blocking)
        speak_async(reply)

    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error: {str(e)}"

    # Save this exchange to memory
    all_history = load_history(1000)
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

.chatbot {
    background: #161622 !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 40px rgba(99,102,241,0.08) !important;
}

.message.user {
    background: #312e81 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #c7d2fe !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

.message.bot {
    background: #1a1a27 !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 14px 14px 14px 4px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

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
# FIX: css passed in gr.Blocks() constructor (not demo.launch) — correct placement
with gr.Blocks(title="Adarsh AI Clone", css=custom_css) as demo:
    gr.Markdown("""
# 🧠 Adarsh AI Clone
**AI Digital Twin** · Manipal University Jaipur · PBL-2 · Local LLM via Ollama
""")

    gr.ChatInterface(
        fn=chat_with_clone,
        chatbot=gr.Chatbot(
            height=460,
            show_label=False,
            avatar_images=(
                None,
                "https://api.dicebear.com/7.x/initials/svg?seed=AS&backgroundColor=6366f1"
            ),
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
    demo.launch(server_name="127.0.0.1", server_port=7860)
