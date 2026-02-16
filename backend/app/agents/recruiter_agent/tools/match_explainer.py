from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

def _normalize_skill(s: str) -> str:
    """Normalise une compétence pour comparaison (lower + strip)."""
    return (s or "").strip().lower()


def analyze_candidate_match(
    candidate_skills: List[str],
    job_requirements: List[str],
    *,
    score_in_percent: bool = True,
) -> Dict[str, Any]:
    """
    Analyse les correspondances entre skills candidat et exigences job.
    """

    candidate_skills = candidate_skills or []
    job_requirements = job_requirements or []

    candidate_norm = {_normalize_skill(s) for s in candidate_skills if _normalize_skill(s)}
    matches: List[str] = []
    gaps: List[str] = []

    for req in job_requirements:
        req_norm = _normalize_skill(req)
        if not req_norm:
            continue
        if req_norm in candidate_norm:
            matches.append(req)
        else:
            gaps.append(req)

    denom = max(len([r for r in job_requirements if _normalize_skill(r)]), 1)
    similarity = len(matches) / denom  # 0..1
    match_score = round(similarity * 100, 2) if score_in_percent else round(similarity, 4)

   
    if matches and gaps:
        explanation = f"Matches on {', '.join(matches)}, but missing {', '.join(gaps)}."
    elif matches and not gaps:
        explanation = f"Matches on {', '.join(matches)}."
    elif not matches and gaps:
        explanation = f"Missing {', '.join(gaps)}."
    else:
        explanation = "No requirements provided."

    return {
        "match_score": match_score,
        "explanation": explanation,
        "matches": matches,
        "gaps": gaps,
        "similarity": round(similarity, 4),
    }


@dataclass
class MatchExplainer:
    """
    Option B: interface orientée classe.
    """
    score_in_percent: bool = True

    def explain(self, candidate_skills: List[str], job_requirements: List[str]) -> Dict[str, Any]:
        return analyze_candidate_match(
            candidate_skills,
            job_requirements,
            score_in_percent=self.score_in_percent,
        )

@tool
def match_explainer_tool(candidate: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explain the match between a candidate and a job description.
    
    Args:
        candidate: A dictionary containing candidate 'skills' (list of strings).
        job: A dictionary containing job 'requirements' or 'skills' (list of strings).
        
    Returns:
        Analysis dictionary (score, matches, gaps).
    """
    candidate_skills = candidate.get("skills") or candidate.get("candidate_skills") or []
    job_requirements = job.get("requirements") or job.get("job_requirements") or job.get("skills") or []

  
    if isinstance(candidate_skills, str):
        candidate_skills = [s.strip() for s in candidate_skills.split(",") if s.strip()]
    if isinstance(job_requirements, str):
        job_requirements = [s.strip() for s in job_requirements.split(",") if s.strip()]

    explainer = MatchExplainer(score_in_percent=True)
    return explainer.explain(candidate_skills=candidate_skills, job_requirements=job_requirements)