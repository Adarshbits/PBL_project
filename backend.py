"""
backend.py — Adarsh AI Clone v9.4
Utility helpers: memory, message building, response cleaning, voice, diagnostics.
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
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:  # FIX: added encoding
            json.dump([], f)
        return True
    except IOError:
        return False


def build_messages(system_prompt: str, current_message: str, n: int = 5) -> list:
    """
    Build a properly formatted Ollama messages list for multi-turn chat.

    KEY FIX for the history bleeding bug:
    Uses Ollama's native role-based message format so previous refusals
    cannot bleed into new unrelated answers.

    Also injects current date/time so the AI always knows what time it is.

    Returns a list of message dicts ready to pass directly to ollama.chat().
    """
    # Inject live date and time — re-computed every call so it's always accurate
    now = datetime.now()
    time_str = now.strftime("%A, %d %B %Y, %I:%M %p")
    time_injection = f"\n\nCurrent date and time: {time_str} IST (India Standard Time)"
    full_system = system_prompt + time_injection

    messages = [{"role": "system", "content": full_system}]

    past = load_history(n)
    for entry in past:
        # FIX: safely unpack — skip malformed/corrupted entries (prevents ValueError crash)
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        user_msg, bot_reply = entry[0], entry[1]
        # FIX: null guard — skip entries with empty or None replies
        if not user_msg or not bot_reply:
            continue
        messages.append({"role": "user",      "content": str(user_msg)})
        messages.append({"role": "assistant", "content": str(bot_reply)})

    # Current user message always goes last
    messages.append({"role": "user", "content": current_message})
    return messages


# ─────────────────────────────────────────────
# IDENTITY LOADER
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
    """
    Load knowledge_base.json and return it as a flat string to inject into
    the system prompt. Expands the AI's dataset beyond the basic identity facts.
    """
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
    """
    Load personality_style.json and format it into prompt instructions.
    Teaches the LLM Adarsh's natural speech patterns, banned corporate AI phrases,
    and tone rules.
    """
    if not os.path.exists(style_file):
        return ""
    try:
        with open(style_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = ["\n--- HOW ADARSH TALKS (STRICT PERSONALITY & TONE) ---"]

        # Banned phrases
        do_not_say = data.get("do_not_say", [])
        if do_not_say:
            lines.append("\nBANNED PHRASES (NEVER say these, they sound like a generic AI bot):")
            for phrase in do_not_say:
                lines.append(f"  - Do NOT say: \"{phrase}\"")

        # Tone rules
        tone_rules = data.get("tone_rules", [])
        if tone_rules:
            lines.append("\nTONE & CONVERSATION RULES:")
            for rule in tone_rules:
                lines.append(f"  - {rule}")

        # Speech patterns
        patterns = data.get("speech_patterns", {})
        if patterns:
            starters = ", ".join(f'"{s}"' for s in patterns.get("sentence_starters", []))
            fillers = ", ".join(f'"{f}"' for f in patterns.get("filler_words", []))
            agree = ", ".join(f'"{a}"' for a in patterns.get("how_to_agree", []))
            lines.append(f"\nNATURAL SPEECH HABITS:")
            if starters:
                lines.append(f"  - Natural starters: {starters}")
            if fillers:
                lines.append(f"  - Casual filler words: {fillers}")
            if agree:
                lines.append(f"  - How to agree: {agree}")

        # Example responses (few-shot learning)
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
    """Build a system prompt string dynamically from identity.json data."""
    if not identity:
        return ""

    name       = identity.get("name", "Adarsh Singh")
    reg        = identity.get("registration_number", "")
    course     = identity.get("course", "")
    semester   = identity.get("semester", "")
    year       = identity.get("year", "")
    university = identity.get("university", "")
    guide      = identity.get("guide", "")
    guide_role = identity.get("guide_role", f"{guide} is your professor and supervisor. You are his student.")
    project    = identity.get("project", "")
    hobbies    = "\n- ".join(identity.get("hobbies", []))
    skills     = ", ".join(identity.get("skills", []))
    style      = identity.get("communication_style", {})
    rules      = identity.get("strict_rules", [])
    rules_text = "\n".join(f"- {r}" for r in rules)

    return f"""You are {name}, a {course} student at {university}.

Facts about you:
- Name: {name}
- Registration Number: {reg}
- Course: {course}, {semester}, {year}
- University: {university}
- Your guide/supervisor: {guide}
- Your role with your guide: {guide_role}
- Project: {project}

Your hobbies (ONLY these, never say anything else):
- {hobbies}

Your skills: {skills}

How you talk:
- Always in English only
- {style.get("length", "Short and direct answers")}
- {style.get("tone", "Slightly friendly tone")}
- Never over-explain

STRICT RULES:
{rules_text}"""


# ─────────────────────────────────────────────
# RESPONSE CLEANING
# ─────────────────────────────────────────────

def clean_reply(raw: str, max_chars: int = 300) -> str:
    """
    Clean LLM output:
    - Strip 'Adarsh:' prefix echoed by model
    - Remove markdown bold/italic markers
    - Truncate cleanly at sentence boundary
    - Remove Hindi/Devanagari characters (force English output)
    """
    text = raw.strip()

    # Remove 'Adarsh:' prefix if model echoes it
    text = re.sub(r"^[Aa]darsh\s*:\s*", "", text)

    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)

    # Strip Devanagari Unicode range (Hindi characters)
    text = re.sub(r"[\u0900-\u097F]+", "", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text).strip()

    # Truncate at sentence boundary within max_chars
    if len(text) > max_chars:
        cutoff = text[:max_chars].rfind(".")
        text = text[:cutoff + 1] if cutoff > 80 else text[:max_chars].rstrip() + "…"

    return text if text else "Ask me anything — about my project, hobbies, or CSE stuff."


# ─────────────────────────────────────────────
# VOICE (async, non-blocking via Edge-TTS)
# ─────────────────────────────────────────────

def speak_async(text: str, max_chars: int = 200) -> None:
    """Play TTS audio in a background thread using Microsoft Edge TTS (Fast & Realistic)."""
    def _speak():
        try:
            import asyncio
            import edge_tts
            from playsound import playsound
            import os
            import time
            import re

            safe_text = re.sub(r"[^\x00-\x7F]+", "", text)[:max_chars]  # ASCII only for TTS
            if not safe_text.strip():
                return

            filename = f"voice_{int(time.time() * 1000)}.mp3"
            
            # Use Edge TTS with a natural Indian male voice
            voice = "en-IN-PrabhatNeural"
            
            async def generate_and_play():
                communicate = edge_tts.Communicate(safe_text, voice)
                await communicate.save(filename)
                
                if os.path.exists(filename):
                    playsound(filename)
                    try:
                        os.remove(filename)
                    except OSError:
                        pass
            
            # Run async function in this thread
            asyncio.run(generate_and_play())

        except ImportError:
            print("[backend] edge-tts or playsound not installed. Skipping voice.")
        except Exception as e:
            print(f"[backend] Voice error: {e}")

    threading.Thread(target=_speak, daemon=True).start()


# ─────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────

def check_ollama(model: str = "llama3.2:1b") -> dict:
    """
    Check if Ollama is running and the model is available.
    FIX: handles both dict-style and object-style SDK responses (version safe).
    Returns dict with 'ok' bool and 'message' string.
    """
    try:
        import ollama
        models = ollama.list()
        # FIX: Ollama SDK returns objects in newer versions, not plain dicts
        model_list = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        available = []
        for m in model_list:
            if isinstance(m, dict):
                available.append(m.get("name", m.get("model", "")))
            else:
                available.append(getattr(m, "model", getattr(m, "name", "")))

        if model in available:
            return {"ok": True, "message": f"✅ Ollama running | Model '{model}' ready"}
        else:
            return {
                "ok": False,
                "message": (
                    f"⚠️ Ollama running but model '{model}' not found.\n"
                    f"Available: {available}\n"
                    f"Fix: ollama pull {model}"
                ),
            }
    except Exception as e:
        return {
            "ok": False,
            "message": (
                f"❌ Ollama not reachable: {e}\n"
                "Fix: run `ollama serve` in a terminal."
            ),
        }


def history_stats() -> dict:
    """Return basic stats about stored chat history."""
    hist = load_history(1000)
    total = len(hist)
    # FIX: null guard — safely check reply exists and is a string before .strip()
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
    print("=== backend.py self-test ===\n")

    print("[1] clean_reply tests")
    print(clean_reply("Adarsh: मैं Adarsh हूँ।  Hello! I'm Adarsh Singh."))
    print(clean_reply("**Hello**, my name is *Adarsh*."))
    print(clean_reply(""))

    print("\n[2] History stats")
    print(history_stats())

    print("\n[3] Ollama check")
    result = check_ollama()
    print(result["message"])

    print("\n[4] build_messages test")
    msgs = build_messages("You are Adarsh.", "What is your hobby?", n=3)
    for m in msgs:
        print(f"  [{m['role']}] {m['content'][:80]}")
