"""
app.py — Adarsh AI Clone v9.7
Main entry point. Chat UI + voice input mic.
All memory, voice, identity logic lives in backend.py.
"""

import gradio as gr
import numpy as np
import io
import ollama

# Import everything from backend
from backend import (
    load_history,
    save_history,
    speak_async,
    clean_reply,
    load_identity,
    load_knowledge_base,
    load_personality_style,
    build_system_prompt,
    build_messages,
)

print("Adarsh AI Clone v9.7 - llama3.2:3b + Personality + Edge-TTS")

# ===================== IDENTITY + KNOWLEDGE + PERSONALITY =====================
_identity = load_identity("identity.json")
_knowledge = load_knowledge_base("knowledge_base.json")
_personality = load_personality_style("personality_style.json")

if _identity:
    SYSTEM_PROMPT = build_system_prompt(_identity) + _knowledge + _personality
    print("Identity + Knowledge Base + Personality Style loaded.")
else:
    SYSTEM_PROMPT = """You are Adarsh Singh, a CSE student at Manipal University Jaipur.
Facts about you:
- Name: Adarsh Singh
- Registration Number: 2427030325
- Course: CSE, 5th Semester, 3rd Year
- University: Manipal University Jaipur
- Your guide/supervisor: Mr. Virendra Mehgwal (he is YOUR professor who guides YOU — you are HIS student, not the other way around)
- Project: AI Digital Twin using Ollama, Gradio, gTTS, JSON memory

Your hobbies (ONLY these):
- Judo, volleyball, football, listening to music

Your skills: Python, AI/ML basics, debugging models

STRICT RULES:
- Answer ONLY what is asked right now
- Mr. Virendra Mehgwal is your guide, you are his student
- Never over-explain
- English only"""
    print("WARNING: identity.json not found -- using hardcoded fallback")

# ===================== AVATAR (CHARACTER ZONE) =====================
IDLE_GIF = "avatar_idle.gif"
TALKING_GIF = "avatar_talking.gif"


def avatar_html(speaking: bool) -> str:
    """Returns the Character Zone markup, pointing at the talking or idle GIF."""
    src = TALKING_GIF if speaking else IDLE_GIF
    status = "Speaking..." if speaking else "Online — llama3.2:3b"
    return f"""
    <div id="character-zone">
        <img src="file={src}" class="avatar-sprite" alt="Adarsh AI avatar" />
        <div class="clone-name">Adarsh AI</div>
        <div class="status-pill">● {status}</div>
    </div>
    """


# ===================== TRANSCRIPTION (MIC) =====================
def transcribe_audio(audio_data):
    """
    Convert recorded mic audio to text using Google Speech Recognition.
    audio_data is a (sample_rate, numpy_array) tuple from gr.Audio.
    Returns transcribed text string or empty string on failure.
    """
    if audio_data is None:
        return ""
    try:
        import speech_recognition as sr
        import scipy.io.wavfile as wav_io

        sample_rate, audio_array = audio_data
        # Ensure it's 16-bit PCM mono
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]  # take left channel if stereo
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

        # Write to in-memory WAV buffer
        wav_buffer = io.BytesIO()
        wav_io.write(wav_buffer, sample_rate, audio_array)
        wav_buffer.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except ImportError:
        return "[Install SpeechRecognition: pip install SpeechRecognition scipy]"
    except Exception as e:
        name = type(e).__name__
        if "UnknownValue" in name:
            return "[Could not understand audio — speak clearly and try again]"
        if "Request" in name:
            return "[No internet for speech recognition — check your connection]"
        return f"[Transcription error: {e}]"

# ===================== MAIN CHAT (STREAMING) =====================
def chat_with_clone(message, history):
    """
    Streaming chat function — text appears word-by-word as Ollama generates.
    Voice plays AFTER the full reply is ready.
    Yields (cleared_input, updated_history, avatar_state) on each chunk.
    """
    if not message or not message.strip():
        yield "", history, avatar_html(False)
        return

    # Add user message to chat immediately
    user_msg = message.strip()
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]

    try:
        # Build proper multi-turn messages with role-based history
        messages = build_messages(SYSTEM_PROMPT, user_msg, n=5)

        # Stream response — text appears word by word
        raw_reply = ""
        stream = ollama.chat(
            model="llama3.2:3b",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            # Handle both object and dict SDK styles
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            elif isinstance(chunk, dict):
                token = chunk.get("message", {}).get("content", "")
            else:
                token = ""
            raw_reply += token
            # Update the last assistant message with partial text
            history[-1]["content"] = raw_reply
            yield "", history, avatar_html(True)

        # Final cleaning after full response is assembled
        reply = clean_reply(raw_reply)
        history[-1]["content"] = reply
        yield "", history, avatar_html(False)

        # Voice plays AFTER text is fully visible
        speak_async(reply)

    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error connecting to Ollama: {str(e)}"
        history[-1]["content"] = reply
        yield "", history, avatar_html(False)

    # Save to persistent memory
    all_history = load_history(1000)
    all_history.append([user_msg, history[-1]["content"]])
    save_history(all_history)


def voice_send(audio_data, history):
    """
    Transcribe mic audio -> send to chat -> return updated history + transcribed text.
    Non-streaming version for mic input (streaming not needed here since
    the user already waited for transcription).
    """
    transcribed = transcribe_audio(audio_data)

    # If transcription failed or returned an error tag, show message but don't send to AI
    if not transcribed or transcribed.startswith("["):
        return transcribed, history, avatar_html(False)

    # For mic input, use non-streaming path for simplicity
    user_msg = transcribed.strip()
    try:
        messages = build_messages(SYSTEM_PROMPT, user_msg, n=5)
        response = ollama.chat(
            model="llama3.2:3b",
            messages=messages,
        )
        if hasattr(response, "message"):
            reply = response.message.content
        elif isinstance(response, dict):
            reply = response.get("message", {}).get("content", "")
        else:
            reply = str(response)
        reply = clean_reply(reply)
        speak_async(reply)
    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error connecting to Ollama: {str(e)}"

    # Save to persistent memory
    all_history = load_history(1000)
    all_history.append([user_msg, reply])
    save_history(all_history)

    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    return transcribed, history, avatar_html(False)

# ===================== CUSTOM CSS =====================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-void: #0a0a0f;
    --bg-panel: #12121a;
    --bg-panel-2: #161622;
    --border-soft: #2d2d3d;
    --accent: #6366f1;
    --accent-2: #22d3ee;
    --accent-glow: rgba(99,102,241,0.4);
    --text-main: #e2e8f0;
    --text-dim: #64748b;
}

body, .gradio-container {
    background: radial-gradient(circle at 15% -10%, #1a1830 0%, var(--bg-void) 55%) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-main) !important;
}

.gradio-container h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0 !important;
}

.gradio-container .prose p {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
}

/* ---------- Character Zone ---------- */
#character-zone {
    background: linear-gradient(160deg, var(--bg-panel-2), var(--bg-panel));
    border: 1px solid var(--border-soft);
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    min-height: 420px;
    position: relative;
    overflow: hidden;
}
#character-zone::before {
    content: "";
    position: absolute;
    width: 220px; height: 220px;
    background: radial-gradient(circle, var(--accent-glow), transparent 70%);
    filter: blur(12px);
    animation: pulse-glow 3.2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { transform: scale(0.9); opacity: 0.55; }
    50% { transform: scale(1.15); opacity: 1; }
}
#character-zone .avatar-sprite {
    width: 160px; height: 160px;
    image-rendering: pixelated;
    border-radius: 50%;
    background: var(--bg-panel);
    border: 2px solid var(--accent);
    box-shadow: 0 0 34px var(--accent-glow);
    z-index: 1;
    object-fit: cover;
}
#character-zone .clone-name {
    margin-top: 16px;
    font-weight: 600;
    font-size: 1.05rem;
    z-index: 1;
}
#character-zone .status-pill {
    margin-top: 6px;
    font-size: 0.72rem;
    color: var(--accent-2);
    border: 1px solid var(--accent-2);
    padding: 3px 12px;
    border-radius: 999px;
    z-index: 1;
    background: rgba(34,211,238,0.08);
}

/* ---------- Chat ---------- */
.chatbot {
    background: var(--bg-panel-2) !important;
    border: 1px solid var(--border-soft) !important;
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
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px 14px 14px 4px !important;
    color: var(--text-main) !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}

/* ---------- Inputs ---------- */
textarea {
    background: #1e1e2a !important;
    color: var(--text-main) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 10px !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    outline: none !important;
}

button.primary {
    background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
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

.mic-section {
    background: var(--bg-panel-2) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-top: 8px !important;
}
.transcribed-box textarea {
    color: var(--accent-2) !important;
    font-style: italic !important;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
"""

# ===================== UI =====================
with gr.Blocks(title="Adarsh AI Clone") as demo:
    gr.Markdown("""
    # Adarsh AI Clone
    **AI Digital Twin** · Manipal University Jaipur · PBL-2 · Local LLM via Ollama
    """)

    with gr.Row():
        # ── Character Zone (left) ───────────────────────────
        with gr.Column(scale=1, min_width=220):
            avatar_box = gr.HTML(avatar_html(False))

        # ── Chat area (right) ───────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=[],
                height=420,
                show_label=False,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/initials/svg?seed=AS&backgroundColor=6366f1"
                ),
            )

            # ── Text input row ───────────────────────────────
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    lines=1,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            # ── Quick example buttons ─────────────────────────
            gr.Examples(
                examples=[
                    "Tell me about yourself",
                    "What are your hobbies?",
                    "Who is your project guide?",
                    "What is your registration number?",
                    "Explain your AI project",
                ],
                inputs=msg_box,
            )

    # ── Voice / Mic input section ───────────────────────────
    gr.Markdown("---\n### Voice Input (Mic)")
    gr.Markdown(
        "_Record your question below. It will be transcribed and sent automatically._",
    )
    with gr.Row():
        mic_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Hold to Record",
        )
        transcribed_box = gr.Textbox(
            label="What I heard:",
            placeholder="Your spoken words will appear here after recording...",
            interactive=False,
            elem_classes=["transcribed-box"],
        )

    # ── Footer ────────────────────────────────────────────────
    gr.Markdown("""
    <div style="text-align:center; color:#475569; font-size:0.75rem; margin-top:16px;">
    Powered by Ollama · llama3.2:3b · Edge-TTS Voice · JSON Memory · SpeechRecognition Mic
    </div>
    """)

    # ── Event wiring ────────────────────────────────────────
    # Text send — via button click
    send_btn.click(
        fn=chat_with_clone,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, avatar_box],
    )

    # Text send — via Enter key
    msg_box.submit(
        fn=chat_with_clone,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, avatar_box],
    )

    # Voice send — fires when audio recording is submitted/changed
    mic_input.change(
        fn=voice_send,
        inputs=[mic_input, chatbot],
        outputs=[transcribed_box, chatbot, avatar_box],
    )

# ===================== RUN =====================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", css=custom_css, allowed_paths=["."])
