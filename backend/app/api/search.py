from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
import sys

from app.agents.recruiter_agent.tools.job_fetcher import job_search_tool

router = APIRouter()

# ── Load embedding model once ────────────────────────────────
_embed_model = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"[Search] Embedding model not available: {e}", file=sys.stderr)


class SearchQuery(BaseModel):
    query: str
    sources: List[str] = ["Remote OK", "Arbeitnow", "We Work Remotely", "The Muse",
                          "Remotive", "Jobicy", "Emploi.tn"]
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


# ═══════════════════════════════════════════════════════════════
#  CV-Based Job Recommendation
# ═══════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    max_results: int = 20
    sources: List[str] = ["Remote OK", "Arbeitnow", "We Work Remotely", "The Muse",
                          "Remotive", "Himalayas", "Emploi.tn"]
    location: Optional[str] = None
    remote_only: bool = False


def _compute_cv_job_compatibility(cv_text: str, cv_skills: list[str],
                                   job_title: str, job_desc: str) -> float:
    """
    Compute compatibility % between a CV and a job using:
      60% embedding similarity + 40% skill/keyword overlap.
    """
    # --- Skill overlap (40%) ---
    cv_skills_lower = {s.lower() for s in cv_skills}
    job_text_lower = f"{job_title} {job_desc}".lower()
    if cv_skills_lower:
        matched = sum(1 for s in cv_skills_lower if s in job_text_lower)
        skill_score = min((matched / max(len(cv_skills_lower), 1)) * 100, 100)
    else:
        skill_score = 0

    # --- Embedding similarity (60%) ---
    embed_score = 0
    if _embed_model is not None:
        try:
            cv_snippet = " ".join(cv_skills[:20]) + " " + cv_text[:500]
            job_snippet = f"{job_title}. {job_desc[:300]}"
            embeddings = _embed_model.encode(
                [cv_snippet, job_snippet], normalize_embeddings=True
            )
            sim = cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1),
            )[0][0]
            embed_score = float(sim) * 100
        except Exception:
            embed_score = skill_score  # fallback

    final = embed_score * 0.6 + skill_score * 0.4
    return round(min(final, 99), 1)


@router.post("/recommend")
async def recommend_jobs(req: RecommendRequest):
    """
    Recommend jobs based on the most recently uploaded CV.
    Uses embeddings + skill overlap (60/40) for compatibility scoring.
    """
    # 1. Get the cached CV
    try:
        from app.api.candidates import get_cv_cache
        cv_cache = get_cv_cache()
    except Exception:
        cv_cache = {}

    if not cv_cache:
        raise HTTPException(
            status_code=400,
            detail="No CV uploaded yet. Please upload a CV first via the Analyze page."
        )

    # Use the most recent CV
    last_key = list(cv_cache.keys())[-1]
    cached = cv_cache[last_key]
    cv_text = cached.get("text", "")
    skills_data = cached.get("skills_data", {})
    cv_skills = skills_data.get("skills", []) if isinstance(skills_data, dict) else []
    job_title_hint = skills_data.get("job_title", "") if isinstance(skills_data, dict) else ""

    # 2. Build a search query from CV skills
    top_skills = cv_skills[:8]
    search_query = " ".join(top_skills) if top_skills else (job_title_hint or "developer engineer")

    # 3. Fetch jobs from all sources
    try:
        result = job_search_tool.invoke({
            "query": search_query,
            "sources": ", ".join(req.sources),
            "max_results": req.max_results * 4,  # over-fetch for filtering
        })
        jobs = result.get("jobs", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job fetch error: {str(e)}")

    # 4. Apply location / remote filters
    if req.remote_only:
        jobs = [j for j in jobs if "remote" in j.get("location", "").lower()
                or "remote" in j.get("title", "").lower()]

    if req.location:
        jobs = [j for j in jobs if _matches_location(j.get("location", ""), req.location)]

    # 5. Compute compatibility for each job
    for job in jobs:
        job["compatibility"] = _compute_cv_job_compatibility(
            cv_text, cv_skills,
            job.get("title", ""), job.get("description", "")
        )

    # 6. Sort by compatibility, take top N
    jobs.sort(key=lambda x: x.get("compatibility", 0), reverse=True)
    jobs = jobs[:req.max_results]

    return {
        "cv_filename": last_key,
        "cv_skills": cv_skills[:15],
        "search_query_used": search_query,
        "total_found": len(jobs),
        "jobs": jobs,
    }
