# main.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union

# Import your modules
from llm_router import route_query_to_wiki  # ← NEW: use the full router
from wikimedia import get_full_wikipedia_extract  # ← rename/adjust as needed

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

# Union response: either one page or many options
IjliyaResponse = Union[SingleResult, DisambiguationResult]

@app.post("/ask")
async def ask_wikipedia(request: QuestionRequest) -> IjliyaResponse:
    user_question = request.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ✅ NEW: Use the full pipeline (topic + disambiguation)
    result = await route_query_to_wiki(user_question)

    # Case 1: No candidates found
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["message"])

    # Case 2: Disambiguation needed
    if result.get("disambiguation"):
        return DisambiguationResult(
            topic=result["topic"],
            options=[
                DisambiguationOption(
                    title=opt["title"],
                    description=opt["description"],
                    url=opt["url"]
                )
                for opt in result["options"]
            ]
        )

    # Case 3: Single match → fetch full extract
    else:
        full_data = get_full_wikipedia_extract(result["title"])
        return SingleResult(
            title=full_data["title"],
            url=full_data["url"],
            extract=full_data.get("extract"),
            source=full_data["source"]
        )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Ijliya"}
