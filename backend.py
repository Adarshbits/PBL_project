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
    Instead of stuffing raw history text into the system prompt (which caused
    previous refusals to bleed into new unrelated answers), we use Ollama's
    native role-based message format: system / user / assistant / user ...

    The model then correctly treats each turn as separate context, so a drug
    refusal on day 1 will NOT contaminate an Iran-Israel question on day 2.

    Returns a list of message dicts ready to pass directly to ollama.chat().
    """
    messages = [{"role": "system", "content": system_prompt}]

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
- Guide: {guide}
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
# VOICE (async, non-blocking)
# ─────────────────────────────────────────────

def speak_async(text: str, max_chars: int = 200) -> None:
    """Play TTS audio in a background thread using gTTS."""
    def _speak():
        try:
            from gtts import gTTS
            from playsound import playsound

            safe_text = re.sub(r"[^\x00-\x7F]+", "", text)[:max_chars]  # ASCII only for TTS
            if not safe_text.strip():
                return

            filename = f"voice_{int(time.time() * 1000)}.mp3"
            tts = gTTS(text=safe_text, lang="en", tld="com.au")
            tts.save(filename)
            playsound(filename)

            try:
                os.remove(filename)
            except OSError:
                pass

        except ImportError:
            print("[backend] gTTS or playsound not installed. Skipping voice.")
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
