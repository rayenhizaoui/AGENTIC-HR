from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re

from app.agents.recruiter_agent.tools.job_fetcher import job_search_tool

router = APIRouter()


class SearchQuery(BaseModel):
    query: str
    sources: List[str] = ["Remote OK", "Arbeitnow", "We Work Remotely", "The Muse"]
    max_results: int = 20
    location: Optional[str] = None
    remote_only: bool = False
    experience_level: Optional[str] = None  # junior, mid, senior


# ── Experience-level keyword heuristics ──────────────────────
_LEVEL_KEYWORDS = {
    "junior": ["junior", "jr", "entry", "intern", "trainee", "graduate", "associé", "débutant"],
    "mid": ["mid", "intermediate", "regular", "confirmed", "confirmé"],
    "senior": ["senior", "sr", "lead", "principal", "staff", "architect", "manager", "head", "director"],
}


def _matches_level(title: str, level: str) -> bool:
    """Check if a job title matches an experience level."""
    if not level or level == "any":
        return True
    t = title.lower()
    for kw in _LEVEL_KEYWORDS.get(level.lower(), []):
        if kw in t:
            return True
    # If level is junior, also accept titles with NO level keywords at all
    if level.lower() == "junior":
        has_senior = any(kw in t for kw in _LEVEL_KEYWORDS["senior"])
        return not has_senior
    return False


def _matches_location(job_location: str, filter_location: str) -> bool:
    """Smart location matching — case-insensitive, partial match."""
    if not filter_location:
        return True
    loc = job_location.lower()
    filt = filter_location.strip().lower()
    # Split filter on commas / spaces for multi-word matching
    parts = [p.strip() for p in re.split(r'[,/]+', filt) if p.strip()]
    return any(p in loc for p in parts) or "remote" in loc


@router.post("/")
async def search_jobs(query: SearchQuery):
    try:
        result = job_search_tool.invoke({
            "query": query.query,
            "sources": ", ".join(query.sources),
            "max_results": query.max_results * 3,  # over-fetch to allow filtering
        })

        jobs = result.get("jobs", [])

        # ── Apply post-fetch filters ─────────────────────────
        if query.remote_only:
            jobs = [
                j for j in jobs
                if "remote" in j.get("location", "").lower()
                or "remote" in j.get("title", "").lower()
            ]

        if query.location:
            jobs = [j for j in jobs if _matches_location(j.get("location", ""), query.location)]

        if query.experience_level and query.experience_level != "any":
            jobs = [j for j in jobs if _matches_level(j.get("title", ""), query.experience_level)]

        # Trim to requested max
        jobs = jobs[: query.max_results]

        return {
            "query": result.get("query", query.query),
            "keywords_used": result.get("keywords_used", []),
            "total_found": len(jobs),
            "sources_queried": ", ".join(query.sources),
            "filters_applied": {
                "location": query.location,
                "remote_only": query.remote_only,
                "experience_level": query.experience_level,
            },
            "jobs": jobs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
