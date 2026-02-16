"""
Candidate Ranking Tools

This module contains the cv_ranker tool for ranking a list
of candidates against a job description.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool

# Import the canonical similarity matcher (single source of truth)
from .similarity_matcher_tool import similarity_matcher_tool


@tool
def cv_ranker(candidates: List[Dict], job_description: str) -> List[Dict]:
    """
    Rank a list of candidates based on similarity to job description.

    Args:
        candidates: List of candidate dicts, each with 'skills', 'experience', 'education'.
        job_description: The job description text to match against.

    Returns:
        Sorted list of candidate dicts with added 'score' field.
    """
    ranked = []
    for cand in candidates:
        res = similarity_matcher_tool.invoke({
            "candidate_profile": cand,
            "job_description": job_description
        })

        score = 0
        if isinstance(res, dict):
            score = res.get("similarity_score", 0)

        cand["score"] = score
        ranked.append(cand)

    return sorted(ranked, key=lambda x: x["score"], reverse=True)
