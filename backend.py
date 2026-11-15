"""
backend.py — Adarsh AI Clone v10.8
Native Windows SAPI5 Speech Engine with accurate status tracking,
identity loaders, conversation memory, response cleaning, Ollama diagnostics.
Imported by app.py. All logic lives here — app.py stays clean.
"""

import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
import pythoncom
import win32com.client

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


def get_initial_history(n: int = 10) -> list:
    """Load past conversations formatted for Gradio Chatbot (role-based dicts)."""
    raw = load_history(n)
    formatted = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            u, a = str(entry[0]).strip(), str(entry[1]).strip()
            if u and a and not a.lower().startswith("error connecting to ollama"):
                formatted.append({"role": "user", "content": u})
                formatted.append({"role": "assistant", "content": a})
    return formatted


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
        user_msg, bot_reply = str(entry[0]).strip(), str(entry[1]).strip()
        if (
            not user_msg
            or not bot_reply
            or bot_reply.lower().startswith("ollama error")
            or bot_reply.lower().startswith("error connecting to ollama")
            or "[winerror" in bot_reply.lower()
        ):
            continue
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": bot_reply})

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
                lines.append(f"  Adarsh: {a}")

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

def clean_reply(raw: str, max_chars: int = 2500) -> str:
    """
    Clean LLM output for display:
    - Strip prefixes ('Adarsh:', 'Adarsh Singh:', 'Assistant:', 'AI:')
    - Strip surrounding quotes ('"..."' or '\'...\'')
    - Remove markdown bold/italic/headers
    - Convert bullet lists to natural sentences
    - Remove Hindi/Devanagari
    - Preserve full conversational responses without premature truncation
    """
    text = raw.strip()

    # Strip leading quotes
    text = text.lstrip('"\' ')

    # Strip conversational role prefixes
    text = re.sub(r"^(?:Adarsh(?:\s+Singh)?|Assistant|AI)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # Strip quotes that may have followed the prefix
    text = text.lstrip('"\' ')

    # Strip trailing quotes
    text = text.rstrip('"\' ')

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)

    # Convert numbered lists "1. item" -> inline text
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

    # Convert bullet points "- item" or "* item" -> inline
    text = re.sub(r"^[-*•]\s+", "", text, flags=re.MULTILINE)

    # Strip Devanagari Unicode range
    text = re.sub(r"[\u0900-\u097F]+", "", text)

    # Collapse multiple newlines and spaces
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()

    # Final trim of quotes
    text = text.strip('"\' ')

    # Truncate cleanly at sentence boundary only if exceeding safety limit
    if len(text) > max_chars:
        cutoff = text[:max_chars].rfind(".")
        text = text[:cutoff + 1] if cutoff > 80 else text[:max_chars].rstrip() + "..."

    return text if text else "Ask me anything."


def clean_for_tts(text: str) -> str:
    """
    Cleaning specifically for text-to-speech.
    Removes symbols, markdown, and formatting to produce clean spoken English.
    """
    t = text
    t = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\d+\.\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^[-*•]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"[^\x00-\x7F]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("...", ".")
    return t


# ─────────────────────────────────────────────
# SAPI5 VOICE ENGINE — Dedicated Thread Worker
# Source of truth: spk.Status.RunningState == 2
# ─────────────────────────────────────────────

class _SAPI5Worker:
    """
    Dedicated COM-initialized SAPI5 worker thread.
    Manages speech requests and cancellation safely without cross-thread COM deadlocks.
    Accurately tracks active hardware speaking status.
    """
    def __init__(self):
        self._cmd_queue = queue.Queue()
        self._is_speaking = False
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()

    def _run(self):
        pythoncom.CoInitialize()
        try:
            spk = win32com.client.Dispatch("SAPI.SpVoice")
            voices = spk.GetVoices()
            for i in range(voices.Count):
                desc = voices.Item(i).GetDescription().lower()
                if "david" in desc or "male" in desc:
                    spk.Voice = voices.Item(i)
                    break
            spk.Rate = 1
        except Exception as e:
            print(f"[backend] SAPI5 init error: {e}")
            pythoncom.CoUninitialize()
            return

        while True:
            cmd, text = self._cmd_queue.get()
            if cmd == "stop":
                try:
                    spk.Speak("", 1 | 2)  # SVSFlagsAsync | SVSFPurgeBeforeSpeak
                except Exception:
                    pass
                with self._lock:
                    self._is_speaking = False
            elif cmd == "speak":
                try:
                    # SVSFlagsAsync = 1, SVSFPurgeBeforeSpeak = 2
                    spk.Speak(text, 1 | 2)

                    # Latch until hardware playback actually starts (RunningState == 2)
                    t0 = time.time()
                    while spk.Status.RunningState != 2 and time.time() - t0 < 0.25:
                        time.sleep(0.005)

                    if spk.Status.RunningState == 2:
                        with self._lock:
                            self._is_speaking = True

                        # Actively poll hardware speaking state (2 = SPRS_IS_SPEAKING)
                        while spk.Status.RunningState == 2:
                            if not self._cmd_queue.empty():
                                break
                            time.sleep(0.02)
                except Exception as e:
                    print(f"[backend] SAPI5 speak error: {e}")
                finally:
                    if self._cmd_queue.empty():
                        with self._lock:
                            self._is_speaking = False

    def speak(self, text: str):
        if not text or not text.strip():
            return
        self._cmd_queue.put(("speak", text.strip()))

    def stop(self):
        with self._lock:
            self._is_speaking = False
        self._cmd_queue.put(("stop", ""))

    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking


_sapi_worker = _SAPI5Worker()


def speak_async(text: str) -> None:
    """Trigger SAPI5 speech asynchronously with zero buffering delay."""
    safe_text = clean_for_tts(text)
    if safe_text:
        _sapi_worker.speak(safe_text)


def stop_speaking() -> None:
    """Halt speech immediately and reset speaking status."""
    _sapi_worker.stop()


def is_speaking() -> bool:
    """Return True if SAPI5 is actively outputting audio."""
    return _sapi_worker.is_speaking()


# ─────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────

def check_ollama(model: str = "llama3.2:3b") -> dict:
    """Check if Ollama is running and the model is available."""
    try:
        import ollama
        models = ollama.list()
        model_list = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        available = []
        for m in model_list:
            if isinstance(m, dict):
                name = m.get("name", m.get("model", ""))
            else:
                name = getattr(m, "model", getattr(m, "name", ""))
            if name:
                available.append(name)

        base_model = model.split(":")[0]
        matched = any(model in a or (base_model in a and "3b" in a) for a in available)

        if matched or model in available:
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


if __name__ == "__main__":
    print("=== backend.py self-test ===")
    print("Testing clean_reply:", clean_reply("Adarsh: Hello there!"))
    print("Testing history stats:", history_stats())
    print("Testing Ollama:", check_ollama()["message"])
