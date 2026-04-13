import gradio as gr
import json
import os
import ollama
import threading
import time

print("🚀 Adarsh AI Clone v9.1 - FIXED VERSION")

# ===================== PERSONALITY =====================
PERSONALITY = """
Name: Adarsh Singh
Course: CSE, 4th Semester
University: Manipal University Jaipur
Registration Number: 2427030325
Guide: Mr. Virendra Mehgal

Communication style:
- Direct but slightly jolly
- Uses simple English
- Avoids unnecessary words

Skills:
- Python
- AI/ML basics
- Debugging models

Hobbies:
- Primary: judo, volleyball
- Secondary: football
- Non-sport: listening to music

Rules:
- Always mention specific hobbies (no generic terms)
- Do not say "outdoor sports"

Project:
- AI Digital Twin using Ollama
- Uses memory (JSON)
- Uses voice (gTTS)
- Uses Gradio UI

Approach:
- Solve problems step by step
- Focus on working prototypes

Behavior:
- Gives short and accurate answers
- Does not over-explain unless asked
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
            json.dump(data, f)
    except:
        pass

# ===================== VOICE =====================
def speak_async(text):
    def _speak():
        try:
            from gtts import gTTS
            from playsound import playsound
            import time

            filename = f"voice_{int(time.time()*1000)}.mp3"
            short_text = text[:200]

            tts = gTTS(text=short_text, lang='en', tld='com.au')
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
    time.sleep(0.3)

    past_history = load_history()[-5:]

    # Only include exchanges that have actual replies
    chat_context = "\n".join(
        [f"U: {h[0]}\nA: {h[1]}" for h in past_history[-3:] if h[1].strip()]
    )

    prompt = f"""You are Adarsh Singh (CSE student MUJ).

RULES:
- Respond in English only
- Answer clearly but briefly
- Do not add unnecessary info
- Use short, natural sentences
- NEVER repeat or mention previous chat history in your reply
- NEVER say things like "as I mentioned" or reference past answers
- Just answer the current question directly

SPECIAL RULES:
- If question is about hobbies -> mention only hobbies
- If follow-up -> continue context silently

PERSONALITY:
{PERSONALITY}

[BACKGROUND CONTEXT - do NOT mention this in your reply]:
{chat_context if chat_context else "No prior context."}

USER: {message}
Adarsh:"""

    try:
        response = ollama.chat(
            model='llama3.2:1b',   # ✅ correct model
            messages=[{'role': 'user', 'content': prompt}]
        )

        print("FULL RESPONSE:", response)  # 🔥 DEBUG

        # ✅ FIXED parsing
        reply = response.get('message', {}).get('content') or response.get('content', '')

        if not reply:
            reply = "⚠️ Empty response from model"

        # Clean reply (safe)
        reply = reply.split("\n")[0]
        reply = reply[:250]

        # Remove intro repetition
        if "my name is" in reply.lower():
            reply = reply.split(".")[-1].strip()

        speak_async(reply)

    except Exception as e:
        print("🔥 OLLAMA ERROR:", str(e))
        reply = f"❌ Error: {str(e)}"

    # Save memory
    all_history = load_history()
    all_history.append([message, reply])
    save_history(all_history)

    return reply

# ===================== UI =====================
demo = gr.ChatInterface(
    fn=chat_with_clone,
    title="🧑‍💻 Adarsh AI Clone v9.1",
    description="AI Digital Twin | Controlled Responses | Voice | Local LLM",
    examples=[
        "Tell me about yourself",
        "What are your hobbies?",
        "Any other hobbies?",
        "How do you debug ML models?"
    ]
)

# ===================== RUN =====================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)