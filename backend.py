"""
backend.py — Adarsh AI Clone v9.2
Utility helpers: memory management, response cleaning, voice, diagnostics.
Imported by app.py.
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
    """Append a [user, reply] pair and save to JSON."""
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
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
        return True
    except IOError:
        return False


def build_context(n: int = 3) -> str:
    """Return recent conversation turns as a formatted string for prompt injection."""
    past = load_history(n)
    lines = []
    for user_msg, bot_reply in past:
        if bot_reply:  # skip empty replies
            lines.append(f"User: {user_msg}")
            lines.append(f"Adarsh: {bot_reply}")
    return "\n".join(lines)


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
    Returns dict with 'ok' bool and 'message' string.
    """
    try:
        import ollama
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
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
    empty = sum(1 for _, r in hist if not r.strip())
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
