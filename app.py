"""
app.py — Adarsh AI Clone v9.5
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
    build_system_prompt,
    build_messages,
)

print("Adarsh AI Clone v9.5 - Mic + Knowledge Base")

# ===================== IDENTITY + KNOWLEDGE =====================
_identity = load_identity("identity.json")
_knowledge = load_knowledge_base("knowledge_base.json")

if _identity:
    SYSTEM_PROMPT = build_system_prompt(_identity) + _knowledge
    print("Identity + Knowledge Base loaded.")
else:
    SYSTEM_PROMPT = """You are Adarsh Singh, a CSE student at Manipal University Jaipur.

Facts about you:
- Name: Adarsh Singh
- Registration Number: 2427030325
- Course: CSE, 4th Semester, 2nd Year
- University: Manipal University Jaipur
- Your guide/supervisor: Mr. Virendra Mehgal (he is YOUR professor who guides YOU — you are HIS student, not the other way around)
- Project: AI Digital Twin using Ollama, Gradio, gTTS, JSON memory

Your hobbies (ONLY these):
- Judo, volleyball, football, listening to music

Your skills: Python, AI/ML basics, debugging models

STRICT RULES:
- Answer ONLY what is asked right now
- Mr. Virendra Mehgal is your guide, you are his student
- Never over-explain
- English only"""
    print("WARNING: identity.json not found -- using hardcoded fallback")


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


# ===================== MAIN CHAT =====================
def chat_with_clone(message, history):
    """
    Core chat function — handles a text message, returns bot reply.
    history is a list of {"role": ..., "content": ...} dicts (Gradio 6.0 format).
    """
    if not message or not message.strip():
        return "", history

    try:
        # Build proper multi-turn messages with role-based history
        messages = build_messages(SYSTEM_PROMPT, message.strip(), n=5)

        response = ollama.chat(
            model="llama3.2:1b",
            messages=messages
        )

        # Parse response — handles both object and dict SDK styles
        if hasattr(response, "message"):
            reply = response.message.content
        elif isinstance(response, dict):
            reply = response.get("message", {}).get("content", "")
        else:
            reply = str(response)

        reply = clean_reply(reply, max_chars=300)
        speak_async(reply)

    except Exception as e:
        print("OLLAMA ERROR:", str(e))
        reply = f"Error connecting to Ollama: {str(e)}"

    # Save to persistent memory
    all_history = load_history(1000)
    all_history.append([message.strip(), reply])
    save_history(all_history)

    # Gradio 6.0 format: list of dicts with role/content keys
    history = history + [
        {"role": "user",      "content": message.strip()},
        {"role": "assistant", "content": reply},
    ]
    return "", history


def voice_send(audio_data, history):
    """
    Transcribe mic audio -> send to chat -> return updated history + transcribed text.
    history is in Gradio 6.0 dict format (list of role/content dicts).
    """
    transcribed = transcribe_audio(audio_data)

    # If transcription failed or returned an error tag, show message but don't send to AI
    if not transcribed or transcribed.startswith("["):
        return transcribed, history

    # Send transcribed text through normal chat pipeline
    _, history = chat_with_clone(transcribed, history)
    return transcribed, history


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

.mic-section {
    background: #161622 !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-top: 8px !important;
}

.transcribed-box textarea {
    color: #22d3ee !important;
    font-style: italic !important;
}
"""

# ===================== UI =====================
with gr.Blocks(title="Adarsh AI Clone") as demo:

    gr.Markdown("""
# Adarsh AI Clone
**AI Digital Twin** · Manipal University Jaipur · PBL-2 · Local LLM via Ollama
""")

    # ── Chat area ──────────────────────────────────────────
    chatbot = gr.Chatbot(
        value=[],
        height=420,
        show_label=False,
        type="messages",   # Gradio 6.0 — use role/content dict format
        avatar_images=(
            None,
            "https://api.dicebear.com/7.x/initials/svg?seed=AS&backgroundColor=6366f1"
        ),
    )

    # ── Text input row ──────────────────────────────────────
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            lines=1,
            scale=9,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    # ── Quick example buttons ───────────────────────────────
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

    # ── Footer ──────────────────────────────────────────────
    gr.Markdown("""
<div style="text-align:center; color:#475569; font-size:0.75rem; margin-top:16px;">
Powered by Ollama · llama3.2:1b · gTTS Voice · JSON Memory · SpeechRecognition Mic
</div>
""")

    # ── Event wiring ────────────────────────────────────────

    # Text send — via button click
    send_btn.click(
        fn=chat_with_clone,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    )

    # Text send — via Enter key
    msg_box.submit(
        fn=chat_with_clone,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    )

    # Voice send — fires when audio recording is submitted/changed
    mic_input.change(
        fn=voice_send,
        inputs=[mic_input, chatbot],
        outputs=[transcribed_box, chatbot],
    )


# ===================== RUN =====================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)
