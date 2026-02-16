"""
LLM-Enhanced Candidate Ranking

Uses Mistral (via Ollama) for feature extraction + sentence-transformers
for semantic scoring. Produces a composite compatibility score (%).

Fallback: if Ollama is unreachable, uses the existing similarity_matcher_tool.
"""

import json
import sys
from typing import Optional

from langchain_core.tools import tool

# ── ML imports (reuse project's existing model) ──────────────
_embed_model = None
_cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cs
    import numpy as np

    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    _cosine_similarity = _cs
except Exception as e:
    print(f"⚠️ SentenceTransformer unavailable for LLM ranker: {e}", file=sys.stderr)

# ── HTTP for Ollama ──────────────────────────────────────────
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


# ═══════════════════════════════════════════════════════════════
#  Ollama / Mistral helpers
# ═══════════════════════════════════════════════════════════════

def _ollama_available() -> bool:
    """Quick health check on Ollama server."""
    if not _HAS_REQUESTS:
        return False
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def query_mistral(prompt: str, max_tokens: int = 512) -> str:
    """
    Send a prompt to local Mistral via Ollama.

    Args:
        prompt: Text prompt.
        max_tokens: Max tokens to generate.

    Returns:
        Generated text, or empty string on failure.
    """
    if not _HAS_REQUESTS:
        return ""

    try:
        resp = _requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": max_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"[Ollama] Error: {e}", file=sys.stderr)
        return ""


def extract_features_llm(text: str, text_type: str = "cv") -> dict:
    """
    Use Mistral to extract structured features from a CV or job description.

    Args:
        text: Raw text of the CV or job description.
        text_type: Either "cv" or "job".

    Returns:
        Dict with keys: skills, experience_years, education, domain, key_requirements.
    """
    prompt = f"""Analyze this {text_type} and extract the following in JSON format.
    IMPORTANT: Ignore general company description ('About Us', 'Mission'). Focus ONLY on the specific ROLE REQUIREMENTS, TECHNICAL SKILLS, and QUALIFICATIONS.

    Output format:
    {{
        "skills": ["list of technical and professional skills found in the text"],
        "experience_years": <number or 0 if unknown>,
        "education": "highest degree or certification requirement",
        "domain": "primary professional domain",
        "key_requirements": ["top 5 specific requirements"]
    }}

    {text_type.upper()} TEXT:
    {text[:8000]}

    Respond ONLY with valid JSON. No explanation, no markdown fences."""

    raw = query_mistral(prompt)
    if not raw:
        return _empty_features()

    try:
        # Find JSON object in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return _empty_features()
        return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return _empty_features()


def _empty_features() -> dict:
    return {
        "skills": [],
        "experience_years": 0,
        "education": "",
        "domain": "",
        "key_requirements": [],
    }


# ═══════════════════════════════════════════════════════════════
#  Composite Ranking
# ═══════════════════════════════════════════════════════════════

def _semantic_score(text_a: str, text_b: str) -> float:
    """
    Compute similarity between two texts.
    Prioritizes sentence-transformers (embeddings).
    Falls back to Jaccard similarity of unique words if embeddings unavailable.
    """
    # 1. Try Embeddings
    if _embed_model and _cosine_similarity:
        try:
            embeddings = _embed_model.encode(
                [text_a, text_b], normalize_embeddings=True
            )
            return float(_cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1),
            )[0][0])
        except Exception as e:
            print(f"⚠️ Embeddings failed: {e}", file=sys.stderr)
            pass

    # 2. Fallback: Jaccard Similarity (Set overlap of words)
    # Simple tokenization by splitting on whitespace and cleaning
    def _tokens(text):
        import re
        words = re.findall(r'\w+', text.lower())
        # Filter out common stop words strictly if needed, but for fallback
        # just length filtering is okay to avoid 'a', 'the' domination
        return {w for w in words if len(w) > 3}

    set_a = _tokens(text_a)
    set_b = _tokens(text_b)
    
    if not set_a or not set_b:
        return 0.0
        
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def _skill_overlap_score(job_skills: list, cv_skills: list) -> tuple[float, list, list]:
    """
    Compute skill overlap ratio.

    Returns:
        (score_0_to_1, matched_skills, missing_skills)
    """
    job_set = {s.lower().strip() for s in job_skills if s}
    cv_set = {s.lower().strip() for s in cv_skills if s}

    if not job_set:
        return 0.0, [], []

    matched = sorted(job_set & cv_set)
    missing = sorted(job_set - cv_set)
    score = len(matched) / len(job_set)
    return score, matched, missing


def rank_single_candidate(
    job_description: str,
    cv_text: str,
    candidate_name: str = "Candidate",
    use_llm: bool = True,
) -> dict:
    """
    Score a single candidate against a job description.

    Args:
        job_description: Full JD text.
        cv_text: Full CV text.
        candidate_name: Display name.
        use_llm: Whether to use Mistral for feature extraction.

    Returns:
        Dict with score, semantic_similarity, skill_match, matched/missing skills.
    """
    # 1. Semantic similarity (Embeddings or Jaccard fallback)
    sem_score = _semantic_score(job_description, cv_text)

    # 2. LLM feature extraction (if available)
    skill_score = 0.0
    matched = []
    missing = []
    cv_features = _empty_features()
    job_features = _empty_features()

    if use_llm and _ollama_available():
        job_features = extract_features_llm(job_description, "job")
        cv_features = extract_features_llm(cv_text, "cv")
        skill_score, matched, missing = _skill_overlap_score(
            job_features.get("skills", []),
            cv_features.get("skills", []),
        )
    else:
        # Better fallback: extract actual skill phrases instead of single words
        _KNOWN_SKILLS = [
            "machine learning", "deep learning", "artificial intelligence", "nlp",
            "computer vision", "pytorch", "tensorflow", "keras", "scikit-learn",
            "pandas", "numpy", "python", "java", "c++", "sql", "nosql",
            "html", "css", "javascript", "typescript", "react", "angular", "vue",
            "node.js", "express", "flask", "django", "fastapi", "spring boot",
            "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "git",
            "data science", "data scientist", "data engineer", "data analyst",
            "rag", "langchain", "transformers", "llms", "prompt engineering",
            "agile", "scrum", "project management", "communication", "leadership",
            "rest api", "graphql", "microservices", "ci/cd", "linux",
            "mlflow", "grafana", "power bi", "tableau", "excel",
            "r", "dax", "spark", "hadoop", "mongodb", "postgresql", "mysql",
            "next.js", ".net", "flutter", "react native", "ios", "android",
            "credit risk", "market risk", "finance", "compliance", "investment",
        ]
        jd_lower = job_description.lower()
        cv_lower = cv_text.lower()

        jd_skills = {s for s in _KNOWN_SKILLS if s in jd_lower}
        cv_skills = {s for s in _KNOWN_SKILLS if s in cv_lower}

        if jd_skills:
            matched = sorted(jd_skills & cv_skills)
            missing = sorted(jd_skills - cv_skills)
            skill_score = len(matched) / len(jd_skills) if jd_skills else 0.0
        else:
            matched, missing = [], []
            skill_score = 0.0

    # 3. Composite score: 60% semantic + 40% skill match
    # Only if matched skills exist, otherwise penalize heavily
    if not matched and not sem_score:
        final = 0.0
    else:
        final = round((sem_score * 0.6 + skill_score * 0.4) * 100, 1)

    return {
        "candidate": candidate_name,
        "score": final,
        "semantic_similarity": round(sem_score * 100, 1),
        "skill_match": round(skill_score * 100, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "cv_features": cv_features,
        "job_features": job_features,
        "llm_used": use_llm and _ollama_available(),
    }


# ═══════════════════════════════════════════════════════════════
#  LangChain Tool
# ═══════════════════════════════════════════════════════════════

@tool
def llm_rank_candidates(
    job_description: str,
    candidate_texts: str,
) -> dict:
    """
    Rank multiple candidates against a job description using LLM + embeddings.

    Use this for deep analysis with Mistral. The user might say:
    "Use LLM to rank candidates", "Deep rank with Mistral", "Analyze compatibility"

    Args:
        job_description: The full job description text.
        candidate_texts: Candidates as a semicolon-separated string.
            Each entry: "Name::CV text". Example: "Alice::Python dev 5y;Bob::Java dev 3y"

    Returns:
        Dict with 'rankings' sorted by score (highest first) and metadata.
    """
    # Parse candidates
    candidates = []
    for entry in candidate_texts.split(";"):
        entry = entry.strip()
        if "::" in entry:
            name, text = entry.split("::", 1)
            candidates.append({"name": name.strip(), "text": text.strip()})
        elif entry:
            candidates.append({"name": f"Candidate {len(candidates)+1}", "text": entry})

    if not candidates:
        return {"error": "No candidates provided.", "rankings": []}

    ollama_up = _ollama_available()

    results = []
    for cand in candidates:
        result = rank_single_candidate(
            job_description=job_description,
            cv_text=cand["text"],
            candidate_name=cand["name"],
            use_llm=ollama_up,
        )
        results.append(result)

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "job_description_preview": job_description[:200] + "...",
        "total_candidates": len(results),
        "llm_used": ollama_up,
        "model": OLLAMA_MODEL if ollama_up else "embeddings-only (Ollama offline)",
        "rankings": results,
    }
