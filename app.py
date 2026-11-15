"""
app.py — Adarsh AI Clone v10.7
Main entry point. Chat UI + voice input mic + guaranteed idle-to-talking avatar animation.
All memory, voice, identity logic lives in backend.py.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Prepend project root to PATH so bundled ffmpeg.exe and ffprobe.exe are found by pydub & Gradio
if BASE_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = BASE_DIR + os.pathsep + os.environ.get("PATH", "")

# Configure pydub explicitly with bundled ffmpeg/ffprobe
try:
    from pydub import AudioSegment
    ffmpeg_bin = os.path.join(BASE_DIR, "ffmpeg.exe")
    ffprobe_bin = os.path.join(BASE_DIR, "ffprobe.exe")
    if os.path.exists(ffmpeg_bin):
        AudioSegment.converter = ffmpeg_bin
    if os.path.exists(ffprobe_bin):
        AudioSegment.ffprobe = ffprobe_bin
except Exception:
    pass

import gradio as gr
import numpy as np
import io
import base64
import time
import re
import threading
import ollama

# Import everything from backend
from backend import (
    load_history,
    get_initial_history,
    save_history,
    speak_async,
    stop_speaking,
    is_speaking,
    clean_reply,
    load_identity,
    load_knowledge_base,
    load_personality_style,
    build_system_prompt,
    build_messages,
    check_ollama,
)

MODEL_NAME = "llama3.2:3b"
print(f"Adarsh AI Clone v10.7 - {MODEL_NAME} + Guaranteed Animated Avatar Transition")

# Startup Ollama Diagnostic Check
_ollama_status = check_ollama(MODEL_NAME)
if _ollama_status["ok"]:
    print(f"[Ollama Status] {_ollama_status['message']}")
else:
    print(f"[Ollama Warning] {_ollama_status['message']}")

# ===================== IDENTITY + KNOWLEDGE + PERSONALITY =====================
_identity = load_identity(os.path.join(BASE_DIR, "identity.json"))
_knowledge = load_knowledge_base(os.path.join(BASE_DIR, "knowledge_base.json"))
_personality = load_personality_style(os.path.join(BASE_DIR, "personality_style.json"))

if _identity:
    SYSTEM_PROMPT = build_system_prompt(_identity) + _knowledge + _personality
    print("Identity + Knowledge Base + Personality Style loaded.")
else:
    SYSTEM_PROMPT = """You are Adarsh Singh, a real 3rd Year B.Tech CSE Core student at Manipal University Jaipur, talking with someone.

YOUR IDENTITY:
- Name: Adarsh Singh
- Reg No: 2427030325
- Course: B.Tech CSE Core, 5th Semester, 3rd Year
- University: Manipal University Jaipur
- Project guide: Dr. Virendra Mehgwal (he is your professor and supervisor)
- PBL-2 Project: AI Digital Twin using Ollama, Gradio, Edge-TTS and JSON memory
- Hobbies: Judo, Volleyball, Football, Listening to music
- Skills: Python, AI/ML basics, debugging models

HOW YOU BEHAVE:
You are a smart, curious, knowledgeable college student. Talk naturally and casually. You can discuss any topic confidently.
Never say "As an AI language model", "Certainly!", or sound robotic. Write in natural flowing sentences."""
    print("WARNING: identity.json not found -- using hardcoded fallback")

# ===================== AVATAR (CHARACTER ZONE) =====================
IDLE_AVATAR_PATH = os.path.join(BASE_DIR, "new_avatar_idle.png")
if not os.path.exists(IDLE_AVATAR_PATH):
    IDLE_AVATAR_PATH = os.path.join(BASE_DIR, "avatar_idle.png")

TALKING_AVATAR_PATH = os.path.join(BASE_DIR, "new_avatar_talking.png")
if not os.path.exists(TALKING_AVATAR_PATH):
    TALKING_AVATAR_PATH = os.path.join(BASE_DIR, "avatar_talking.png")


def load_image_as_data_uri(path):
    try:
        if not path or not os.path.exists(path):
            return None
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext == "png":
            mime = "image/png"
        elif ext in ("jpg", "jpeg"):
            mime = "image/jpeg"
        elif ext == "webp":
            mime = "image/webp"
        else:
            mime = "image/gif"
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"WARNING: avatar load error at {path}: {e}")
        return None


IDLE_AVATAR_DATA = load_image_as_data_uri(IDLE_AVATAR_PATH)
TALKING_AVATAR_DATA = load_image_as_data_uri(TALKING_AVATAR_PATH)


def avatar_html(speaking: bool) -> str:
    """
    Renders Character Zone markup.
    Guaranteed transition:
      - When speaking is True: renders active talking container with mouth flap keyframe animation
        and glowing 'Speaking...' status pill.
      - When speaking is False: renders idle avatar with mouth closed and 'Online' badge.
    """
    status = "Speaking..." if speaking else f"Online — {MODEL_NAME}"
    speak_class = " speaking" if speaking else ""

    if IDLE_AVATAR_DATA and TALKING_AVATAR_DATA:
        if speaking:
            avatar_inner = f"""
            <div class="avatar-container speaking">
                <img src="{IDLE_AVATAR_DATA}" class="avatar-layer avatar-idle-img" alt="Adarsh AI idle" />
                <img src="{TALKING_AVATAR_DATA}" class="avatar-layer avatar-talking-img" alt="Adarsh AI talking" />
            </div>
            <style>
              .avatar-container.speaking .avatar-talking-img {{
                  animation: mouth-flap 0.30s steps(1, end) infinite !important;
              }}
              @keyframes mouth-flap {{
                  0%, 48% {{ opacity: 0; }}
                  50%, 98% {{ opacity: 1; }}
              }}
            </style>
            """
        else:
            avatar_inner = f"""
            <div class="avatar-container">
                <img src="{IDLE_AVATAR_DATA}" class="avatar-layer avatar-idle-img" alt="Adarsh AI idle" />
            </div>
            """
    elif IDLE_AVATAR_DATA:
        avatar_inner = f"""
        <div class="avatar-container{speak_class}">
            <img src="{IDLE_AVATAR_DATA}" class="avatar-layer" alt="Adarsh AI avatar" />
        </div>
        """
    else:
        avatar_inner = '<div class="avatar-container"><div class="avatar-fallback">🤖</div></div>'

    return f"""
    <div id="character-zone">
        {avatar_inner}
        <div class="clone-name">Adarsh AI</div>
        <div class="status-pill{speak_class}">● {status}</div>
    </div>
    """


# ===================== TRANSCRIPTION (MIC) =====================
def transcribe_audio(audio_data):
    """
    Convert recorded mic audio to text using Google Speech Recognition.
    """
    if audio_data is None:
        return ""
    try:
        import speech_recognition as sr
        import scipy.io.wavfile as wav_io
        from pydub import AudioSegment

        recognizer = sr.Recognizer()

        if isinstance(audio_data, tuple):
            sample_rate, audio_array = audio_data
            if len(audio_array) == 0:
                return ""
            if audio_array.ndim > 1:
                audio_array = audio_array[:, 0]  # mono

            if audio_array.dtype == np.int32:
                audio_array = (audio_array >> 16).astype(np.int16)
            elif np.issubdtype(audio_array.dtype, np.floating):
                audio_array = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            elif audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)

            max_amp = int(np.max(np.abs(audio_array)))
            if 0 < max_amp < 6000:
                gain = min(4.0, 20000.0 / max_amp)
                audio_array = np.clip(audio_array.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

            wav_buffer = io.BytesIO()
            wav_io.write(wav_buffer, sample_rate, audio_array)
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio = recognizer.record(source)

        elif isinstance(audio_data, dict) and "path" in audio_data:
            path = audio_data["path"]
            if not os.path.exists(path):
                return ""
            seg = AudioSegment.from_file(path)
            wav_buffer = io.BytesIO()
            seg.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio = recognizer.record(source)

        elif isinstance(audio_data, str) and os.path.exists(audio_data):
            seg = AudioSegment.from_file(audio_data)
            wav_buffer = io.BytesIO()
            seg.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio = recognizer.record(source)

        else:
            return ""

        text = recognizer.recognize_google(audio)
        return text.strip()
    except Exception as e:
        name = type(e).__name__
        err_msg = str(e)
        if "UnknownValue" in name:
            return "[Could not understand audio — please speak clearly and try again]"
        if "Request" in name or "WinError 623" in err_msg or "Failed to establish" in err_msg:
            return "[Microphone STT requires internet for Google Speech API — please check connection or type in chat]"
        return f"[Transcription note: {e}]"


# ===================== MAIN CHAT (STREAMING + GUARANTEED AVATAR SHIFT) =====================
def chat_with_clone(message, history):
    """
    Streaming chat function — text appears word-by-word as Ollama generates.
    Avatar state is 100% synchronized with native SAPI5 playback status:
    transitions to talking when playback starts, remains talking for the full
    duration of speech, and returns to idle as soon as speech finishes or is cancelled.
    """
    if not message or not message.strip():
        yield "", history, avatar_html(False)
        return

    user_msg = message.strip()
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]

    try:
        messages = build_messages(SYSTEM_PROMPT, user_msg, n=5)

        raw_reply = ""
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            elif isinstance(chunk, dict):
                token = chunk.get("message", {}).get("content", "")
            else:
                token = ""
            raw_reply += token
            history[-1]["content"] = raw_reply
            yield "", history, avatar_html(False)

        reply = clean_reply(raw_reply)
        history[-1]["content"] = reply

        # 1. Trigger voice playback asynchronously via native SAPI5
        speak_async(reply)

        # 2. Wait briefly for playback to actually start (latches in ~15-50ms)
        t_wait = time.time()
        while not is_speaking() and time.time() - t_wait < 0.35:
            time.sleep(0.01)

        # 3. Synchronize avatar strictly with active SAPI5 playback state
        if is_speaking():
            yield "", history, avatar_html(True)
            while is_speaking():
                time.sleep(0.04)

        # 4. Return to idle stage once playback genuinely ends or is stopped
        yield "", history, avatar_html(False)

    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error connecting to Ollama: {str(e)}"
        history[-1]["content"] = reply
        yield "", history, avatar_html(False)
        return

    final_reply = history[-1]["content"].strip()
    if final_reply and not final_reply.startswith("Error connecting to Ollama"):
        all_history = load_history(1000)
        all_history.append([user_msg, final_reply])
        save_history(all_history)


def voice_send(audio_data, history):
    """
    Transcribe mic audio -> stream reply -> synchronized avatar talking -> idle reset.
    """
    if audio_data is None:
        yield "", history, avatar_html(False)
        return

    transcribed = transcribe_audio(audio_data)

    if not transcribed or transcribed.startswith("["):
        yield transcribed, history, avatar_html(False)
        return

    user_msg = transcribed.strip()

    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    yield transcribed, history, avatar_html(False)

    try:
        messages = build_messages(SYSTEM_PROMPT, user_msg, n=5)

        raw_reply = ""
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            elif isinstance(chunk, dict):
                token = chunk.get("message", {}).get("content", "")
            else:
                token = ""
            raw_reply += token
            history[-1]["content"] = raw_reply
            yield transcribed, history, avatar_html(False)

        reply = clean_reply(raw_reply)
        history[-1]["content"] = reply

        # 1. Trigger voice playback asynchronously via native SAPI5
        speak_async(reply)

        # 2. Wait briefly for playback to actually start (latches in ~15-50ms)
        t_wait = time.time()
        while not is_speaking() and time.time() - t_wait < 0.35:
            time.sleep(0.01)

        # 3. Synchronize avatar strictly with active SAPI5 playback state
        if is_speaking():
            yield transcribed, history, avatar_html(True)
            while is_speaking():
                time.sleep(0.04)

        # 4. Return to idle stage once playback genuinely ends or is stopped
        yield transcribed, history, avatar_html(False)

    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error connecting to Ollama: {str(e)}"
        history[-1]["content"] = reply
        yield transcribed, history, avatar_html(False)
        return

    final_reply = history[-1]["content"].strip()
    if final_reply and not final_reply.startswith("Error connecting to Ollama"):
        all_history = load_history(1000)
        all_history.append([user_msg, final_reply])
        save_history(all_history)


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

.avatar-container {
    position: relative;
    width: 160px;
    height: 160px;
    border-radius: 50%;
    overflow: hidden;
    background: var(--bg-panel);
    border: 2px solid var(--accent);
    box-shadow: 0 0 34px var(--accent-glow);
    z-index: 1;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.avatar-container.speaking {
    border-color: var(--accent-2);
    box-shadow: 0 0 38px rgba(34, 211, 238, 0.55);
    animation: avatar-speak-pulse 1.3s ease-in-out infinite alternate;
}
@keyframes avatar-speak-pulse {
    0% { transform: scale(1); }
    100% { transform: scale(1.035); }
}

.avatar-layer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
}
.avatar-idle-img {
    z-index: 1;
}
.avatar-talking-img {
    z-index: 2;
    opacity: 0;
}
.avatar-container.speaking .avatar-talking-img {
    animation: mouth-flap 0.30s steps(1, end) infinite;
}
@keyframes mouth-flap {
    0%, 48% { opacity: 0; }
    50%, 98% { opacity: 1; }
}

.avatar-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    width: 100%;
    height: 100%;
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
    transition: all 0.2s ease;
}
#character-zone .status-pill.speaking {
    color: #4ade80;
    border-color: #4ade80;
    background: rgba(74, 222, 128, 0.12);
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

button.stop, .stop-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
button.stop:hover, .stop-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
    background: linear-gradient(135deg, #f87171, #ef4444) !important;
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
                value=get_initial_history(10),
                height=420,
                show_label=False,
                avatar_images=(None, IDLE_AVATAR_PATH if os.path.exists(IDLE_AVATAR_PATH) else None),
            )

            # ── Text input row ───────────────────────────────
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    lines=1,
                    scale=8,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1, elem_classes=["stop-btn"])

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
        "_Click the microphone icon to record your question, then click stop (■). It will be transcribed and answered automatically._",
    )
    with gr.Row():
        mic_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Microphone (Record / Stop)",
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
    Powered by Ollama · llama3.2:3b · Edge-TTS Voice · Animated Pixel Avatar · JSON Memory
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

    # Voice send — fires automatically when audio recording is submitted/changed
    mic_input.change(
        fn=voice_send,
        inputs=[mic_input, chatbot],
        outputs=[transcribed_box, chatbot, avatar_box],
    )

    # Stop button — stops audio playback immediately and resets avatar
    def stop_speaking_handler():
        stop_speaking()
        return avatar_html(False)

    stop_btn.click(
        fn=stop_speaking_handler,
        inputs=None,
        outputs=[avatar_box],
    )

# ===================== RUN =====================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", css=custom_css)
