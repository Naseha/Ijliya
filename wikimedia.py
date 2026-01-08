# wikimedia.py
import requests
from typing import Optional, Dict, Any, List
import time
import urllib.parse

# 🌐 Set a valid User-Agent + longer timeout
HEADERS = {
    "User-Agent": "Ijliya/1.0 (https://github.com/yourusername/ijliya; naseha@example.com) Python/requests"
}
TIMEOUT = 30
MAX_RETRIES = 3


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
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                print(f"  ⚠️  HTTP {status} (attempt {attempt + 1}/{MAX_RETRIES + 1}), retrying...")
                time.sleep(1)
                continue
            print(f"  ❌ HTTP error {status}: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            return None
    return None


# ✅ NEW: Get disambiguation options from Wikipedia (opensearch)
def get_wikipedia_disambiguation_options(topic: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Returns list of candidate pages for `topic` using Wikipedia's opensearch.
    Each item: {"title": ..., "description": ..., "url": ...}
    """
    params = {
        "action": "opensearch",
        "search": topic,
        "limit": limit,
        "format": "json",
        "namespace": 0
    }
    data = _get_with_retry("https://en.wikipedia.org/w/api.php", params)
    if not data or len(data) < 4:
        return []

    titles = data[1]
    descs = data[2]
    urls = data[3]

    results = []
    for title, desc, url in zip(titles, descs, urls):
        results.append({
            "title": title,
            #"description": (desc[:197] + "...") if len(desc) > 200 else desc or "No description available.",
            "description": desc if desc else "No description available.",
            "url": url
        })

    for r in results:
    print(f"✅ Title: {r['title']}")
    print(f"📄 Description: {r['description']}")
    print(f"🔗 URL: {r['url']}")
    
    return results


# ✅ NEW: Get full Ijliya response by Wikipedia title (bypass topic search)
def get_ijliya_response_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    Given a Wikipedia page title, fetch its Wikidata QID and full extract.
    """
    # Step 1: Get Wikidata QID from Wikipedia page
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }
    wp_data = _get_with_retry("https://en.wikipedia.org/w/api.php", params)
    if not wp_data:
        return None

    pages = wp_data.get("query", {}).get("pages", {})
    page_id = next(iter(pages)) if pages else None
    if not page_id or page_id == "-1":
        return None

    page = pages[page_id]
    wikibase_item = page.get("pageprops", {}).get("wikibase_item")
    if not wikibase_item:
        # Fallback: try to get extract directly without Wikidata
        return _fallback_extract_by_title(title)

    # Step 2: Now use Wikidata QID to get full extract (your original flow)
    return get_wikipedia_page(wikibase_item)


def _fallback_extract_by_title(title: str) -> Dict[str, Any]:
    """Fallback when no Wikidata item exists."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "format": "json"
    }
    wp_data = _get_with_retry("https://en.wikipedia.org/w/api.php", params)
    if not wp_data:
        return {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "extract": None,
            "source": "Wikipedia (CC BY-SA)"
        }

    pages = wp_data["query"]["pages"]
    page_id = next(iter(pages))
    if page_id == "-1":
        return {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "extract": None,
            "source": "Wikipedia (CC BY-SA)"
        }

    page = pages[page_id]
    extract = page.get("extract", "")
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

    return {
        "title": title,
        "url": url,
        "extract": extract,
        "source": "Wikipedia (CC BY-SA)"
    }


# 🔁 Keep your original functions (unchanged in logic)
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
    params_wp = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "format": "json"
    }
    print(f"📄 Fetching extract for: {title}")
    wp_data = _get_with_retry("https://en.wikipedia.org/w/api.php", params_wp)
    if not wp_data:
        return None

    pages = wp_data["query"]["pages"]
    page_id = next(iter(pages))
    if page_id == "-1":
        return None

    page = pages[page_id]
    extract = page.get("extract", "")
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

    return {
        "title": title,
        "url": url,
        "extract": extract,
        "source": "Wikipedia (CC BY-SA)"
    }


def get_ijliya_response(topic: str) -> Optional[Dict[str, Any]]:
    """Legacy: topic → Wikidata → Wikipedia (used when disambiguation not needed)"""
    wd_result = search_wikidata(topic)
    if not wd_result:
        return None
    return get_wikipedia_page(wd_result["id"])


# 🔍 Test
if __name__ == "__main__":
    test_topic = "Pluto"
    print(f"🧪 Testing disambiguation for: '{test_topic}'\n")
    options = get_wikipedia_disambiguation_options(test_topic, limit=5)
    if options:
        print("📋 Candidates:")
        for i, opt in enumerate(options, 1):
            print(f"{i}. {opt['title']}")
            print(f"   {opt['description']}\n")
    else:
        print("❌ No disambiguation options found.")
