# main.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union

# Import your actual functions
from llm_router import route_query_to_wiki
from wikimedia import get_wikipedia_disambiguation_options, get_ijliya_response_by_title

app = FastAPI(
    title="Ijliya API",
    description="A Wiki-only answer engine. No AI hallucinations — only Wikipedia links.",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# --- Response Models ---
class SingleResult(BaseModel):
    type: str = "single"
    title: str
    url: str
    extract: Optional[str] = None
    source: str = "Wikipedia (CC BY-SA)"

class DisambiguationOption(BaseModel):
    title: str
    description: str
    url: str

class DisambiguationResult(BaseModel):
    type: str = "disambiguation"
    topic: str
    options: List[DisambiguationOption]
    source: str = "Wikipedia (CC BY-SA)"

# Union response type (FastAPI supports this in modern versions)
IjliyaResponse = Union[SingleResult, DisambiguationResult]

@app.post("/ask")
async def ask_wikipedia(request: QuestionRequest) -> IjliyaResponse:
    user_question = request.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1: Extract clean topic using LLMs
    topic = await route_query_to_wiki(user_question)
    if not topic or not isinstance(topic, str):
        topic = user_question

    # Step 2: Get disambiguation candidates from Wikipedia
    candidates = get_wikipedia_disambiguation_options(topic, limit=5)

    # Handle: no candidates
    if not candidates:
        search_url = f"https://en.wikipedia.org/wiki/Special:Search?search={user_question.replace(' ', '%20')}"
        return DisambiguationResult(
            topic=topic,
            options=[DisambiguationOption(
                title="Search Wikipedia",
                description="No exact matches found. Try searching manually.",
                url=search_url
            )],
            source="Wikipedia (CC BY-SA)"
        )

    # Handle: multiple candidates → show disambiguation
    if len(candidates) > 1:
        return DisambiguationResult(
            topic=topic,
            options=[
                DisambiguationOption(
                    title=opt["title"],
                    description=opt["description"],
                    url=opt["url"]
                )
                for opt in candidates
            ]
        )

    # Handle: exactly one candidate → fetch full extract via Wikidata
    else:
        title = candidates[0]["title"]
        full_result = get_ijliya_response_by_title(title)
        if full_result:
            return SingleResult(
                title=full_result["title"],
                url=full_result["url"],
                extract=full_result.get("extract"),
                source=full_result["source"]
            )
        else:
            # Fallback if extract fails
            return SingleResult(
                title=title,
                url=candidates[0]["url"],
                extract=None,
                source="Wikipedia (CC BY-SA)"
            )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Ijliya"}
