# wikimedia.py
import requests
from typing import Optional, Dict, Any
import time

# 🌐 Set a valid User-Agent + longer timeout
HEADERS = {
    "User-Agent": "Ijliya/1.0 (https://github.com/yourusername/ijliya; naseha@example.com) Python/requests"
}
TIMEOUT = 30  # increased from 5
MAX_RETRIES = 3  # retry once on timeout


def _get_with_retry(url: str, params: dict) -> Optional[dict]:
    """Helper to safely fetch from Wikimedia APIs with retry."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"  ⏱️  Timeout (attempt {attempt + 1}/{MAX_RETRIES + 1})")
            if attempt < MAX_RETRIES:
                time.sleep(1)
            else:
                print(f"  ❌ Failed after {MAX_RETRIES + 1} attempts")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # Retry on transient server errors or rate limiting
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                print(f"  ⚠️  HTTP {status} (attempt {attempt + 1}/{MAX_RETRIES + 1}), retrying...")
                time.sleep(1)
                continue
            # For client errors like 400/404, no point retrying
            print(f"  ❌ HTTP error {status}: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            return None
    return None

def search_wikidata(topic: str) -> Optional[Dict[str, Any]]:
    params = {
        "action": "wbsearchentities",
        "search": topic,
        "language": "en",
        "format": "json",
        "limit": 1
    }
    print(f"🔍 Searching Wikidata for: '{topic}'")
    data = _get_with_retry("https://www.wikidata.org/w/api.php", params)
    if data and data.get("search"):
        item = data["search"][0]
        print(f"✅ Found: {item['label']} (QID: {item['id']})")
        return {
            "id": item["id"],
            "label": item["label"],
            "description": item.get("description", "")
        }
    return None


def get_wikipedia_page(qid: str) -> Optional[Dict[str, Any]]:
    # Step 1: Get sitelinks from Wikidata
    params_wd = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "sitelinks",
        "languages": "en",
        "format": "json"
    }
    print(f"🌐 Fetching Wikipedia link for QID: {qid}")
    wd_data = _get_with_retry("https://www.wikidata.org/w/api.php", params_wd)
    if not wd_data:
        return None

    entities = wd_data.get("entities", {})
    if qid not in entities:
        return None

    sitelinks = entities[qid].get("sitelinks", {})
    enwiki = sitelinks.get("enwiki")
    if not enwiki:
        print("⚠️  No English Wikipedia page found")
        return None

    title = enwiki["title"]
    
    # Step 2: Get extract from Wikipedia
    params_wp = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "format": "json"
    }
    print(f"📄 Fetching extract for: {title}")
    #wp_data = _get_with_retry("https://en.wikipedia.org/api.php", params_wp)
    wp_data = _get_with_retry("https://en.wikipedia.org/w/api.php", params_wp)
    
    if not wp_data:
        return None

    pages = wp_data["query"]["pages"]
    page_id = next(iter(pages))
    if page_id == "-1":
        return None

    page = pages[page_id]
    extract = page.get("extract", "")
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    return {
        "title": title,
        "url": url,
        "extract": extract,
        "source": "Wikipedia (CC BY-SA)"
    }


def get_ijliya_response(topic: str) -> Optional[Dict[str, Any]]:
    wd_result = search_wikidata(topic)
    if not wd_result:
        return None
    return get_wikipedia_page(wd_result["id"])


# 🔍 Test
if __name__ == "__main__":
    test_topic = "photosynthesis"
    print(f"🧪 Testing Ijliya with topic: '{test_topic}'\n")
    result = get_ijliya_response(test_topic)
    if result:
        print("\n🎉 FINAL RESULT:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Source: {result['source']}")
        print(f"Extract: {result['extract'][:120]}...")
    else:
        print("\n❌ No result after all retries.")