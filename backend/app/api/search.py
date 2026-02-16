from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.agents.recruiter_agent.tools.job_fetcher import job_search_tool

router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    sources: List[str] = ["Remote OK", "Arbeitnow"]
    max_results: int = 15

@router.post("/")
async def search_jobs(query: SearchQuery):
    try:
        result = job_search_tool.invoke({
            "query": query.query,
            "sources": ", ".join(query.sources),
            "max_results": query.max_results
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
