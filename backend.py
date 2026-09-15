"""
backend.py — Adarsh AI Clone v10.0
Utility helpers: memory, message building, response cleaning, voice (Edge-TTS + pygame cancel), diagnostics.
Imported by app.py. All logic lives here — app.py stays clean.
"""

import json
import os
import re
import threading
import time
from datetime import datetime

HISTORY_FILE = "chat_history.json"

# ─────────────────────────────────────────────
# VOICE CANCELLATION STATE
# A new speak_async() sets _stop_event to kill the current playback.
# ─────────────────────────────────────────────
_stop_event = threading.Event()
_stop_event.set()  # start as "already stopped" so first call works fine




# ─────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────

def load_history(n: int = 10) -> list:
    """Load last n chat exchanges from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[-n:] if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def save_history(data: list) -> bool:
    """Save the full history list to JSON."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"[backend] Save error: {e}")
        return False


def clear_history() -> bool:
    """Wipe chat history file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return True
    except IOError:
        return False


def build_messages(system_prompt: str, current_message: str, n: int = 6) -> list:
    """
    Build a properly formatted Ollama messages list for multi-turn chat.
    Injects current date/time so the AI always knows what time it is.
    """
    now = datetime.now()
    time_str = now.strftime("%A, %d %B %Y, %I:%M %p")
    time_injection = f"\n\nCurrent date and time: {time_str} IST (India Standard Time)"
    full_system = system_prompt + time_injection

    messages = [{"role": "system", "content": full_system}]

    past = load_history(n)
    for entry in past:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        user_msg, bot_reply = entry[0], entry[1]
        if not user_msg or not bot_reply:
            continue
        messages.append({"role": "user",      "content": str(user_msg)})
        messages.append({"role": "assistant", "content": str(bot_reply)})

    messages.append({"role": "user", "content": current_message})
    return messages


# ─────────────────────────────────────────────
# IDENTITY & KNOWLEDGE LOADERS
# ─────────────────────────────────────────────

def load_identity(identity_file: str = "identity.json") -> dict:
    """Load identity data from JSON file. Returns empty dict if not found."""
    if not os.path.exists(identity_file):
        return {}
    try:
        with open(identity_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[backend] Identity load error: {e}")
        return {}


def load_knowledge_base(kb_file: str = "knowledge_base.json") -> str:
    """Load knowledge_base.json and return formatted string for system prompt."""
    if not os.path.exists(kb_file):
        return ""
    try:
        with open(kb_file, "r", encoding="utf-8") as f:
            kb = json.load(f)

        lines = ["\n--- ADDITIONAL KNOWLEDGE ABOUT ADARSH ---"]
        for section, entries in kb.items():
            if section == "faqs":
                lines.append("\nFrequently Asked Questions:")
                for item in entries:
                    if isinstance(item, dict):
                        lines.append(f"  Q: {item.get('q','')}")
                        lines.append(f"  A: {item.get('a','')}")
            elif isinstance(entries, list):
                lines.append(f"\n{section.upper()}:")
                for entry in entries:
                    if isinstance(entry, str):
                        lines.append(f"  - {entry}")
        lines.append("--- END KNOWLEDGE ---")
        return "\n".join(lines)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[backend] Knowledge base load error: {e}")
        return ""


def load_personality_style(style_file: str = "personality_style.json") -> str:
    """Load personality_style.json and format into prompt instructions."""
    if not os.path.exists(style_file):
        return ""
    try:
        with open(style_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = ["\n--- HOW ADARSH TALKS (PERSONALITY & TONE) ---"]

        do_not_say = data.get("do_not_say", [])
        if do_not_say:
            lines.append("\nNEVER say these (they sound like a generic AI bot):")
            for phrase in do_not_say:
                lines.append(f"  - Do NOT say: \"{phrase}\"")

        tone_rules = data.get("tone_rules", [])
        if tone_rules:
            lines.append("\nTONE & CONVERSATION RULES:")
            for rule in tone_rules:
                lines.append(f"  - {rule}")

        patterns = data.get("speech_patterns", {})
        if patterns:
            starters = ", ".join(f'"{s}"' for s in patterns.get("sentence_starters", []))
            agree = ", ".join(f'"{a}"' for a in patterns.get("how_to_agree", []))
            lines.append("\nNATURAL SPEECH HABITS:")
            if starters:
                lines.append(f"  - Natural starters: {starters}")
            if agree:
                lines.append(f"  - How to agree: {agree}")

        examples = data.get("example_responses", {})
        if examples:
            lines.append("\nFEW-SHOT EXAMPLES OF HOW ADARSH ANSWERS:")
            for q, a in examples.items():
                lines.append(f"  Q: \"{q}\"")
                lines.append(f"  Adarsh: \"{a}\"")

        lines.append("--- END PERSONALITY INSTRUCTIONS ---")
        return "\n".join(lines)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[backend] Personality style load error: {e}")
        return ""


def build_system_prompt(identity: dict) -> str:
    """Build a smart, unrestricted system prompt from identity.json."""
    if not identity:
        return ""

    name       = identity.get("name", "Adarsh Singh")
    reg        = identity.get("registration_number", "")
    course     = identity.get("course", "")
    semester   = identity.get("semester", "")
    year       = identity.get("year", "")
    university = identity.get("university", "")
    guide      = identity.get("guide", "")
    guide_role = identity.get("guide_role", f"{guide} is your professor and supervisor.")
    project    = identity.get("project", "")
    hobbies    = ", ".join(identity.get("hobbies", []))
    skills     = ", ".join(identity.get("skills", []))

    return f"""You are {name}, a real {year} {course} student at {university}, talking with someone.

YOUR IDENTITY (never deny or forget this):
- Name: {name}
- Reg No: {reg}
- Course: {course}, {semester}, {year}
- University: {university}
- Project guide: {guide} ({guide_role})
- PBL-2 Project: {project}
- Hobbies: {hobbies}
- Skills: {skills}

HOW YOU BEHAVE — READ THIS CAREFULLY:
You are a smart, curious, real college student. You talk about ANYTHING — world events, technology, science, movies, sports, current affairs, philosophy, whatever. You are NOT a restricted bot. You have opinions. You engage. You are knowledgeable.

When asked about world events or general knowledge: answer confidently using what you know.
When asked about something that happened very recently (today's breaking news you'd have no way of knowing): say "I haven't seen that news, but let me think about it..." and engage anyway.
When asked personal questions about Adarsh: answer from your identity above.

TONE RULES (NON-NEGOTIABLE):
- Talk like a real 21-year-old student — casual, direct, natural
- Keep answers concise unless they need detail
- Use contractions: "I'm", "don't", "it's" — never "I am", "do not"
- NEVER say "Certainly!", "Absolutely!", "Great question!", "As an AI...", "I'd be happy to help!"
- NEVER refuse to engage with a topic — find a way to discuss it
- No bullet point lists in replies — write in natural flowing sentences
- No markdown formatting (no **bold**, no ## headers) — plain conversational text only"""


# ─────────────────────────────────────────────
# RESPONSE CLEANING
# ─────────────────────────────────────────────

def clean_reply(raw: str, max_chars: int = 600) -> str:
    """
    Clean LLM output for display:
    - Strip 'Adarsh:' prefix
    - Remove markdown bold/italic/headers
    - Convert bullet lists to natural sentences
    - Remove Hindi/Devanagari
    - Truncate cleanly at sentence boundary
    """
    text = raw.strip()

    # Remove 'Adarsh:' prefix if model echoes it
    text = re.sub(r"^[Aa]darsh\s*:\s*", "", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)

    # Convert numbered lists "1. item" → inline text
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

    # Convert bullet points "- item" or "* item" → inline
    text = re.sub(r"^[-*•]\s+", "", text, flags=re.MULTILINE)

    # Strip Devanagari Unicode range
    text = re.sub(r"[\u0900-\u097F]+", "", text)

    # Collapse multiple newlines and spaces
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()

    # Truncate at sentence boundary within max_chars
    if len(text) > max_chars:
        cutoff = text[:max_chars].rfind(".")
        text = text[:cutoff + 1] if cutoff > 80 else text[:max_chars].rstrip() + "..."

    return text if text else "Ask me anything."


def clean_for_tts(text: str) -> str:
    """
    Additional cleaning specifically for text-to-speech.
    Converts any remaining list-like structures to natural spoken sentences.
    """
    t = text

    # Remove any leftover markdown
    t = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\d+\.\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^[-*•]\s+", "", t, flags=re.MULTILINE)

    # Strip non-ASCII (TTS handles ASCII best)
    t = re.sub(r"[^\x00-\x7F]+", "", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Remove ellipsis (it confuses TTS)
    t = t.replace("...", ".")

    return t


# ─────────────────────────────────────────────
# VOICE — Edge-TTS (CANCELLABLE via Subprocess)
# ─────────────────────────────────────────────

_current_audio_process = None
_stop_event = threading.Event()

def stop_speaking() -> None:
    """Stop any currently playing speech immediately."""
    global _current_audio_process, _stop_event
    _stop_event.set()
    
    # Kill the subprocess playing the audio
    if _current_audio_process:
        try:
            import psutil
            parent = psutil.Process(_current_audio_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
        except Exception:
            try:
                _current_audio_process.terminate()
            except:
                pass
        _current_audio_process = None


def speak_async(text: str) -> None:
    """
    Play TTS audio in a background thread using Edge-TTS.
    Cancels any previously playing speech first — no more double-speaking.
    """
    global _stop_event, _current_audio_process

    # Cancel whatever is currently speaking
    stop_speaking()

    # Create a fresh stop event for this new speech
    my_event = threading.Event()
    _stop_event = my_event

    def _speak():
        global _current_audio_process
        try:
            import asyncio
            import edge_tts
            import subprocess
            import sys

            safe_text = clean_for_tts(text)
            if not safe_text or my_event.is_set():
                return

            filename = f"voice_{int(time.time() * 1000)}.mp3"
            voice = "en-IN-PrabhatNeural"

            async def generate():
                communicate = edge_tts.Communicate(safe_text, voice)
                await communicate.save(filename)

            asyncio.run(generate())

            # Check if cancelled during generation
            if my_event.is_set() or not os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass
                return

            # Play the audio in a separate Python process so it can be killed
            play_script = f"from playsound import playsound; playsound(r'{filename}')"
            _current_audio_process = subprocess.Popen([sys.executable, "-c", play_script])
            
            # Wait for it to finish naturally
            _current_audio_process.wait()
            
            # Clean up the file
            try:
                os.remove(filename)
            except OSError:
                pass

        except ImportError:
            print("[backend] edge-tts not installed. Skipping voice.")
        except Exception as e:
            print(f"[backend] Voice error: {e}")

    threading.Thread(target=_speak, daemon=True).start()



# ─────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────

def check_ollama(model: str = "llama3.1:8b") -> dict:
    """Check if Ollama is running and the model is available."""
    try:
        import ollama
        models = ollama.list()
        model_list = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        available = []
        for m in model_list:
            if isinstance(m, dict):
                available.append(m.get("name", m.get("model", "")))
            else:
                available.append(getattr(m, "model", getattr(m, "name", "")))

        if model in available:
            return {"ok": True, "message": f"Ollama running | Model '{model}' ready"}
        else:
            return {
                "ok": False,
                "message": (
                    f"Ollama running but model '{model}' not found.\n"
                    f"Available: {available}\n"
                    f"Fix: ollama pull {model}"
                ),
            }
    except Exception as e:
        return {
            "ok": False,
            "message": (
                f"Ollama not reachable: {e}\n"
                "Fix: run `ollama serve` in a terminal."
            ),
        }


def history_stats() -> dict:
    """Return basic stats about stored chat history."""
    hist = load_history(1000)
    total = len(hist)
    empty = sum(
        1 for entry in hist
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and not (entry[1] or "").strip()
    )
    return {
        "total_exchanges": total,
        "empty_replies": empty,
        "file": HISTORY_FILE,
        "exists": os.path.exists(HISTORY_FILE),
    }


# ─────────────────────────────────────────────
# QUICK SELF-TEST (run: python backend.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== backend.py v10.0 self-test ===\n")

    print("[1] clean_reply tests")
    print(clean_reply("Adarsh: My hobbies are:\n1. Judo\n2. Volleyball\n3. Football"))
    print(clean_reply("**Hello**, my name is *Adarsh*."))
    print(clean_reply(""))

    print("\n[2] clean_for_tts test")
    print(clean_for_tts("My hobbies are:\n- Judo\n- Volleyball\n- Football"))

    print("\n[3] History stats")
    print(history_stats())

    print("\n[4] Ollama check")
    result = check_ollama()
    print(result["message"])

    print("\n[5] build_messages test")
    msgs = build_messages("You are Adarsh.", "What is your hobby?", n=3)
    for m in msgs:
        print(f"  [{m['role']}] {m['content'][:80]}")
