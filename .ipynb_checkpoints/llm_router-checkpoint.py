# llm_router.py
import os
import asyncio
from typing import Optional, Tuple
import traceback
import json
from google import genai
import asyncio
from openrouter import OpenRouter
# Import SDKs (install with: pip install groq google-generativeai openrouter-py huggingface_hub requests)
from groq import Groq
#import google.generativeai as genai
from google import genai
#import openrouter_py
from openrouter import OpenRouter
from huggingface_hub import InferenceClient
import requests

from dotenv import load_dotenv
from pathlib import Path

# load .env from project root (optional: pass explicit path)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

import os, sys
print("Python executable:", sys.executable)
print("Working dir:", os.getcwd())
print("GROQ_API_KEY present:", bool(os.getenv("GROQ_API_KEY")))
# preview first 6 chars if present (safe, do not paste full keys anywhere)
print("GROQ_API_KEY preview:", (os.getenv("GROQ_API_KEY") or "")[:6])

DEBUG = True  # set False to silence raw-response prints

# Load API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Primary
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")  # Backup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

def check_keys():
    missing = []
    for name in ("GROQ_API_KEY","GEMINI_API_KEY","OPENROUTER_API_KEY",
                 "HUGGINGFACE_API_KEY","SAMBANOVA_API_KEY","CEREBRAS_API_KEY"):
        if not os.getenv(name):
            missing.append(name)
    if missing:
        print("⚠️ Missing API keys:", ", ".join(missing))

check_keys()


# Prompt template: extract only topic
PROMPT_TEMPLATE = (
    "Extract only the main topic or entity from this question. "
    "Return only the topic name — no punctuation, no explanation, no extra words.\n"
    "Question: {question}\n"
    "Topic:"
)

# Helper: clean output
def clean_output(text: str) -> str:
    return text.strip().split('\n')[0].strip(' ."')

# --- Provider Functions ---
#Grok Working
async def try_groq(question: str) -> Optional[str]:
    """Groq call (no success logging here)."""
    if not GROQ_API_KEY:
        print("❌ Groq skipped: GROQ_API_KEY not set")
        return None

    def call_sync():
        client = Groq(api_key=GROQ_API_KEY)
        return client.chat.completions.create(
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            model="llama-3.1-8b-instant",
            max_tokens=20,
            temperature=0.0,
            top_p=1.0,
            stream=False
        )

    try:
        resp = await asyncio.to_thread(call_sync)

        # safe conversion for debug (use model_dump when available)
        try:
            if hasattr(resp, "model_dump"):
                raw = resp.model_dump()
            elif hasattr(resp, "dict"):
                raw = resp.dict()
            elif hasattr(resp, "to_dict"):
                raw = resp.to_dict()
            else:
                raw = resp
            raw_json = json.dumps(raw, default=lambda o: str(o))[:1000]
        except Exception:
            raw_json = str(resp)[:1000]

        # (Optional) keep this debug print while developing, remove later
        print("🔎 Groq raw response (truncated):", raw_json)

        # extract text
        text = None
        if hasattr(resp, "choices") and resp.choices:
            choice = resp.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content
            else:
                text = getattr(choice, "text", None) or getattr(choice, "content", None)
        elif isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                text = choices[0].get("message", {}).get("content") or choices[0].get("text") or choices[0].get("content")

        return clean_output(text) if text else None

    except Exception as e:
        print("❌ Groq call raised an exception:", str(e)[:400])
        return None

#Hugging Face
async def try_huggingface(question: str) -> Optional[str]:
    if not HUGGINGFACE_API_KEY:
        if DEBUG: print("❌ HuggingFace skipped: HUGGINGFACE_API_KEY not set")
        return None

    def call_sync():
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
        # many HF clients accept inputs= or prompt=; adapt if your version differs
        return client.text_generation(
            model="microsoft/Phi-3-mini-4k-instruct",
            inputs=PROMPT_TEMPLATE.format(question=question),
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False
        )

    try:
        resp = await asyncio.to_thread(call_sync)

        # debug raw
        try:
            raw = resp if isinstance(resp, (dict, list)) else (resp.model_dump() if hasattr(resp, "model_dump") else resp)
            raw_json = json.dumps(raw, default=lambda o: str(o))[:1000]
        except Exception:
            raw_json = str(resp)[:1000]
        if DEBUG: print("🔎 HuggingFace raw response (truncated):", raw_json)

        # extract text
        text = None
        if isinstance(resp, dict):
            text = resp.get("generated_text") or (resp.get("choices") or [{}])[0].get("text")
        elif isinstance(resp, list) and resp:
            text = resp[0].get("generated_text") or resp[0].get("text")
        else:
            # fallback to string
            text = str(resp)

        return clean_output(text) if text else None

    except Exception as e:
        if DEBUG:
            print("❌ HuggingFace exception:")
            traceback.print_exc()
        return None
        

#Gemini 

async def try_gemini(question: str) -> Optional[str]:
    if not GEMINI_API_KEY and not GEMINI_API_KEY_2:
        if DEBUG: print("❌ Gemini skipped: no GEMINI_API_KEY set")
        return None

    keys = [GEMINI_API_KEY, GEMINI_API_KEY_2]
    for i, key in enumerate(keys):
        if not key:
            continue

        def call_sync():
            # create client per key (sync)
            client = genai.Client(api_key=key)
            # SDK shapes vary; pass a minimal request body
            return client.models.generate_content(
                model="gemini-1.5-flash",
                # many SDKs accept a list of contents or a single string; adapt if needed
                contents=[PROMPT_TEMPLATE.format(question=question)],
                max_output_tokens=20,
                temperature=0.0
            )

        try:
            resp = await asyncio.to_thread(call_sync)

            # debug raw
            try:
                raw = resp.model_dump() if hasattr(resp, "model_dump") else (resp.dict() if hasattr(resp, "dict") else resp)
                raw_json = json.dumps(raw, default=lambda o: str(o))[:1000]
            except Exception:
                raw_json = str(resp)[:1000]
            if DEBUG: print(f"🔎 Gemini raw response (truncated): {raw_json}")

            # extract text (common shapes)
            text = None
            if hasattr(resp, "candidates") and getattr(resp, "candidates"):
                cand = resp.candidates[0]
                text = getattr(cand, "content", None) or getattr(cand, "text", None)
            elif hasattr(resp, "text"):
                text = resp.text
            elif isinstance(resp, dict):
                # try common dict shapes
                text = resp.get("text") or (resp.get("candidates") or [{}])[0].get("content") or (resp.get("candidates") or [{}])[0].get("text")

            if text:
                return clean_output(text)

            if DEBUG:
                print(f"❌ Gemini (key {i+1}) returned no text; raw (truncated): {raw_json}")
        except Exception as e:
            if DEBUG:
                print(f"❌ Gemini (key {i+1}) exception:")
                traceback.print_exc()
            continue

    return None
    
#Open Router
async def try_openrouter(question: str) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        if DEBUG: print("❌ OpenRouter skipped: OPENROUTER_API_KEY not set")
        return None

    def call_sync():
        client = OpenRouter(api_key=OPENROUTER_API_KEY)
        return client.chat.completions.create(
            model="minimax/minimax-m2",
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            max_tokens=20,
            temperature=0.0
        )

    try:
        resp = await asyncio.to_thread(call_sync)

        # debug raw
        try:
            raw = resp.model_dump() if hasattr(resp, "model_dump") else (resp.dict() if hasattr(resp, "dict") else resp)
            raw_json = json.dumps(raw, default=lambda o: str(o))[:1000]
        except Exception:
            raw_json = str(resp)[:1000]
        if DEBUG: print("🔎 OpenRouter raw response (truncated):", raw_json)

        # extract text
        text = None
        if hasattr(resp, "choices") and resp.choices:
            choice = resp.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content
            else:
                text = getattr(choice, "text", None) or getattr(choice, "content", None)
        elif isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                text = choices[0].get("message", {}).get("content") or choices[0].get("text") or choices[0].get("content")

        return clean_output(text) if text else None

    except Exception as e:
        if DEBUG:
            print("❌ OpenRouter exception:")
            traceback.print_exc()
        return None

#Sambanova

async def try_sambanova(question: str) -> Optional[str]:
    if not SAMBANOVA_API_KEY:
        if DEBUG: print("❌ SambaNova skipped: SAMBANOVA_API_KEY not set")
        return None

    url = "https://api.sambanova.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "max_tokens": 20,
        "temperature": 0.0
    }

    def call():
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            if DEBUG:
                print("❌ SambaNova HTTP error", resp.status_code, resp.text[:400])
            raise
        return resp.json()

    try:
        data = await asyncio.to_thread(call)

        if DEBUG:
            print("🔎 SambaNova raw response (truncated):", json.dumps(data, default=lambda o: str(o))[:1000])

        text = None
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                text = choices[0].get("message", {}).get("content") or choices[0].get("text") or choices[0].get("content")

        return clean_output(text) if text else None

    except Exception as e:
        if DEBUG:
            print("❌ SambaNova exception:")
            traceback.print_exc()
        return None

 #Cerebras       
async def try_cerebras(question: str) -> Optional[str]:
    if not CEREBRAS_API_KEY:
        if DEBUG: print("❌ Cerebras skipped: CEREBRAS_API_KEY not set")
        return None

    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3.1-8b",
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "max_tokens": 20,
        "temperature": 0.0
    }

    def call():
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            if DEBUG:
                print("❌ Cerebras HTTP error", resp.status_code, resp.text[:400])
            raise
        return resp.json()

    try:
        data = await asyncio.to_thread(call)

        if DEBUG:
            print("🔎 Cerebras raw response (truncated):", json.dumps(data, default=lambda o: str(o))[:1000])

        text = None
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                text = choices[0].get("message", {}).get("content") or choices[0].get("text") or choices[0].get("content")

        return clean_output(text) if text else None

    except Exception as e:
        if DEBUG:
            print("❌ Cerebras exception:")
            traceback.print_exc()
        return None        

# --- Main Router ---

async def extract_topic_fallback(question: str) -> str:
    """Try all 6 providers in sequence until one succeeds."""
    providers = [
        ("Groq", try_groq),
        ("Gemini", try_gemini),
        ("OpenRouter", try_openrouter),
        ("HuggingFace", try_huggingface),
        ("SambaNova", try_sambanova),
        ("Cerebras", try_cerebras)
    ]
    
    print(f"🧠 Extracting topic from: '{question}'")
    
    for name, func in providers:
        print(f"→ Trying {name}...")
        result = await func(question)
        if result:
            print(f"✅ {name} extracted: '{result}'")
            return result
    
    # Final fallback: use raw question
    print("⚠️ All LLMs failed. Using raw query as topic.")
    return question.strip()

# --- Test Function ---
if __name__ == "__main__":
    async def test():
        test_question = "Who was Marie Curie?"
        topic = await extract_topic_fallback(test_question)
        print("\n🎯 Final topic:", repr(topic))
    
    asyncio.run(test())
