# llm_router.py
import os
import asyncio
import requests
from typing import Optional, List, Dict, Any

# Import SDKs
from groq import Groq
import google.generativeai as genai
from openrouter import OpenRouter
from huggingface_hub import InferenceClient

# Load API keys from environment (set in Render dashboard)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# Prompt template: 
prompt = """You are an information retrieval assistant. Your task is to gather factual, verbatim excerpts from Wikipedia and its sister sites (Wikivoyage, Wikibooks, Wikinews, Wikidata, Wikimedia Commons) related to the user's query. Do not create summaries or add any interpretations; only replicate 2-3 lines directly from the relevant Wikipedia sections or sister sites that contain essential details about the topic.

You must only search and extract information from these specified sites. Do not search or include content from any other sources or websites.

If the query refers to multiple topics (e.g., "Pluto" as a planet, Disney character, or mythological figure), provide 2-3 lines about each relevant aspect, clearly indicating each. After presenting the information, ask the user which specific aspect or topic they are interested in to refine the response.

Question: {question}
Your response should include:
- Exact 2-3 lines from Wikipedia or sister sites for each relevant topic.
- No summaries or creative writing.
- Clear indication of different topics if multiple are addressed.
- A polite question asking the user to specify their preferred aspect or topic for more focused information.
"""


# Helper: clean output
def clean_output(text: str) -> str:
    return text.strip().split('\n')[0].strip(' ."')

# --- LLM PROVIDERS (unchanged) ---
async def try_groq(question: str) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            model="llama-3.1-8b-instant",
            max_tokens=20,
            temperature=0.0,
        )
        return clean_output(chat.choices[0].message.content)
    except Exception as e:
        print(f"❌ Groq failed: {str(e)[:60]}")
        return None

async def try_gemini(question: str) -> Optional[str]:
    keys = [GEMINI_API_KEY, GEMINI_API_KEY_2]
    for key in keys:
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await asyncio.to_thread(
                model.generate_content,
                PROMPT_TEMPLATE.format(question=question),
                generation_config={"max_output_tokens": 20, "temperature": 0.0}
            )
            return clean_output(response.text)
        except Exception as e:
            print(f"❌ Gemini failed: {str(e)[:60]}")
            continue
    return None

async def try_openrouter(question: str) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    try:
        client = OpenRouter(api_key=OPENROUTER_API_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="qwen/qwen2.5-1.5b-instruct",
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            max_tokens=20,
            temperature=0.0,
        )
        return clean_output(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ OpenRouter failed: {str(e)[:60]}")
        return None

async def try_huggingface(question: str) -> Optional[str]:
    if not HUGGINGFACE_API_KEY:
        return None
    try:
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
        response = await asyncio.to_thread(
            client.text_generation,
            prompt=PROMPT_TEMPLATE.format(question=question),
            model="microsoft/Phi-3-mini-4k-instruct",
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
        )
        return clean_output(response)
    except Exception as e:
        print(f"❌ HuggingFace failed: {str(e)[:60]}")
        return None

async def try_sambanova(question: str) -> Optional[str]:
    if not SAMBANOVA_API_KEY:
        return None
    try:
        url = "https://api.sambanova.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {SAMBANOVA_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "Meta-Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            "max_tokens": 20,
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return clean_output(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"❌ SambaNova failed: {str(e)[:60]}")
        return None

async def try_cerebras(question: str) -> Optional[str]:
    if not CEREBRAS_API_KEY:
        return None
    try:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            "max_tokens": 20,
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return clean_output(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"❌ Cerebras failed: {str(e)[:60]}")
        return None

# --- WIKIMEDIA DISAMBIGUATION LAYER ---
def get_wikipedia_disambiguation(topic: str, limit: int = 5) -> List[Dict[str, str]]:
    print(f"Fetching disambiguation options for: {topic}")
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": topic,
            "limit": limit,
            "format": "json",
            "namespace": 0,
            "utf8": 1
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        print(f"API response data: {data}")
        if len(data) < 4:
            print("API response data is malformed or empty.")
            return []

        titles = data[1]
        descriptions = data[2]
        urls = data[3]

        results = []
        for title, desc, url in zip(titles, descriptions, urls):
            print(f"Candidate title: {title}")
            print(f"Description: {desc}")
            print(f"URL: {url}")
            results.append({
                "title": title,
                "description": (desc[:197] + "...") if len(desc) > 200 else desc or "No description available.",
                "url": url
            })
        return results
    except Exception as e:
        print(f"Error fetching disambiguation options: {str(e)}")
        return []
        
# --- MAIN ROUTER: Topic extraction + Disambiguation ---
async def route_query_to_wiki(question: str) -> Dict[str, Any]:
    print(f"Received question: {question}")
    # Step 1: Extract clean topic using fallback LLMs
    topic = await extract_topic_fallback(question)
    print(f"Extracted topic: {topic}")

    # Query Wikimedia for candidate pages
    candidates = get_wikipedia_disambiguation(topic, limit=5)
    print(f"Candidates retrieved: {candidates}")

    # Show a few lines from each candidate if available
    for idx, candidate in enumerate(candidates):
        print(f"Candidate {idx+1}: {candidate['title']}")
        print(f"Description: {candidate.get('description', '')}")
        print(f"URL: {candidate.get('url', '')}")
        print("-" * 40)

    # Handle no candidates
    if not candidates:
        return {
            "error": True,
            "message": "No relevant Wikipedia pages found for your query."
        }

    # Handle exactly one candidate
    if len(candidates) == 1:
        title = candidates[0]["title"]
        full_result = get_ijliya_response_by_title(title)
        print(f"Full result for title '{title}': {full_result}")
        if full_result:
            return SingleResult(
                title=full_result["title"],
                url=full_result["url"],
                extract=full_result.get("extract"),
                source=full_result["source"]
            )
        else:
            return SingleResult(
                title=title,
                url=candidates[0]["url"],
                extract=None,
                source="Wikipedia (CC BY-SA)"
            )

    # Multiple candidates
    else:
        return DisambiguationResult(
            topic=topic,
            options=[
                DisambiguationOption(
                    title=opt["title"],
                    description=opt["description"],
                    url=opt["url"]
                )
                for opt in candidates
            ],
            source="Wikipedia (CC BY-SA)"
        )
# --- Legacy: Fallback topic extractor (used internally) ---
async def extract_topic_fallback(question: str) -> str:
    providers = [
        ("Groq", try_groq),
        ("Gemini", try_gemini),
        ("OpenRouter", try_openrouter),
        ("HuggingFace", try_huggingface),
        ("SambaNova", try_sambanova),
        ("Cerebras", try_cerebras),
    ]
    for name, func in providers:
        print(f"Trying provider: {name}")
        result = await func(question)
        print(f"Result from {name}: {result}")
        if result:
            print(f"Using result from {name}: {result}")
            return result
    print("No provider returned a result, defaulting to original question.")
    return question.strip()
