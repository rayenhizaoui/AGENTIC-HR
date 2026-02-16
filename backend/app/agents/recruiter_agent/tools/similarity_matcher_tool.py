"""
Similarity Matcher Tool

Computes semantic similarity between a candidate profile and a job description.
Includes robust fallback to keyword matching if AI models fail to load.
"""

import sys
from langchain_core.tools import tool
from typing import Optional, Dict

# robust import handling for ML libraries
_model = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Initialize model once
    _model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"⚠️ Warning: Could not load SentenceTransformer ({str(e)}). Using keyword-based fallback.", file=sys.stderr)
    _model = None
    cosine_similarity = None


def _candidate_json_to_text(candidate_profile: dict) -> str:
    """
    Convert structured candidate JSON into a single text string.
    """
    sections = []

    skills = candidate_profile.get("skills")
    if isinstance(skills, list) and skills:
        sections.append("Skills: " + ", ".join(skills))

    experience = candidate_profile.get("experience")
    if isinstance(experience, list) and experience:
        # Handle list of strings or list of objects? Assuming strings based on graph.py
        # graph.py passes: [f"{years} years experience"]
        sections.append("Experience: " + ", ".join([str(e) for e in experience]))

    education = candidate_profile.get("education")
    if isinstance(education, str) and education.strip():
        sections.append("Education: " + education)
    elif isinstance(education, list):
         sections.append("Education: " + ", ".join([str(e) for e in education]))

    return ". ".join(sections)


def _keyword_similarity(candidate_text: str, job_description: str) -> float:
    """Fallback ranking logic using simple set overlap of keywords."""
    cand_tokens = set(candidate_text.lower().split())
    job_tokens = set(job_description.lower().split())
    
    if not job_tokens:
        return 0.0
        
    overlap = cand_tokens.intersection(job_tokens)
    # Simple calculation: percentage of job keywords (roughly) matched
    # But job description has many stop words.
    
    # Let's count significant matches only? 
    # For now, simplistic Jaccard * boost
    score = (len(overlap) / len(job_tokens)) * 100 * 3.0 
    return min(95.0, round(score, 2))


@tool
def similarity_matcher_tool(candidate_profile: dict, job_description: str) -> dict:
    """
    Compute semantic similarity score between a candidate profile
    and a job description.
    
    Args:
        candidate_profile: Dict with keys 'skills', 'experience', 'education'
        job_description: String text of the JD
    """
    if not candidate_profile or not job_description:
        return {"similarity_score": 0.0}

    try:
        candidate_text = _candidate_json_to_text(candidate_profile)
        
        # Use Transformer model if available
        if _model:
            embeddings = _model.encode(
                [candidate_text, job_description],
                normalize_embeddings=True
            )

            candidate_embedding = embeddings[0].reshape(1, -1)
            job_embedding = embeddings[1].reshape(1, -1)

            similarity = cosine_similarity(candidate_embedding, job_embedding)[0][0]
            
            return {
                "similarity_score": round(float(similarity) * 100, 2)
            }
        else:
            # Fallback
            score = _keyword_similarity(candidate_text, job_description)
            return {
                "similarity_score": score,
                "note": "Computed using keyword matching fallback (Model unavailable)."
            }
            
    except Exception as e:
        # Catch-all fallback
        candidate_text = _candidate_json_to_text(candidate_profile) if candidate_profile else ""
        score = _keyword_similarity(candidate_text, job_description)
        return {
            "similarity_score": score, 
            "error": str(e),
            "note": "Fallback due to calculation error."
        }
