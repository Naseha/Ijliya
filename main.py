# main.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import your modules
from llm_router import extract_topic_fallback
from wikimedia import get_ijliya_response

# Initialize FastAPI app
app = FastAPI(
    title="Ijliya API",
    description="A Wiki-only answer engine. No AI hallucinations — only Wikipedia links.",
    version="1.0"
)

# Allow requests from your frontend (GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuestionRequest(BaseModel):
    question: str

# Response model
class IjliyaResponse(BaseModel):
    wiki_links: List[str]
    title: str
    extract: Optional[str] = None
    source: str

@app.post("/ask", response_model=IjliyaResponse)
async def ask_wikipedia(request: QuestionRequest):
    """
    Ask a question. Get only Wikipedia/Wikimedia content.
    """
    user_question = request.question.strip()
    
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Step 1: Extract topic using redundant LLM router
    topic = await extract_topic_fallback(user_question)
    
    # Step 2: Fetch from Wikimedia
    wiki_result = get_ijliya_response(topic)
    
    if not wiki_result:
        # Fallback: return search link
        search_url = f"https://en.wikipedia.org/wiki/Special:Search?search={user_question.replace(' ', '%20')}"
        return IjliyaResponse(
            wiki_links=[search_url],
            title="Search Wikipedia",
            extract=None,
            source="Wikipedia (CC BY-SA)"
        )
    
    # Return clean, compliant response
    return IjliyaResponse(
        wiki_links=[wiki_result["url"]],
        title=wiki_result["title"],
        extract=wiki_result.get("extract"),
        source=wiki_result["source"]
    )

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Ijliya"}