"""
Semantic Skill Extractor

Hybrid extraction combining:
    1. Regex keyword matching (deterministic, fast)
    2. Sentence-transformer embeddings (catches synonyms like "ML" → "machine learning")
    3. spaCy NER (soft skills, organizations, products)

This module is designed as an **enhancement layer** on top of `skill_extractor_tool`.
Call `semantic_enhance_skills()` after the regex pass to discover skills that
static keywords missed.
"""

from __future__ import annotations

import re
import sys
from typing import List, Optional, Set

from langchain_core.tools import tool

# ── Lazy-loaded models ────────────────────────────────────────

_sentence_model = None
_spacy_nlp = None


def _get_sentence_model():
    """Lazy-load SentenceTransformer (shared with similarity_matcher_tool)."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _sentence_model
    except Exception as e:
        print(f"⚠️ SentenceTransformer unavailable: {e}", file=sys.stderr)
        return None


def _get_spacy_nlp():
    """Lazy-load spaCy model (en_core_web_sm or fr_core_news_sm)."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        for model_name in ["en_core_web_sm", "fr_core_news_sm", "en_core_web_md"]:
            try:
                _spacy_nlp = spacy.load(model_name)
                print(f"spaCy loaded: {model_name}")
                return _spacy_nlp
            except OSError:
                continue
        print("⚠️ No spaCy model found. NER-based extraction disabled.", file=sys.stderr)
        return None
    except ImportError:
        print("⚠️ spaCy not installed. NER-based extraction disabled.", file=sys.stderr)
        return None


# ── Canonical skill list for embedding comparison ────────────

# These are the "ground truth" skills against which we compare CV phrases.
# Grouped for readability; flattened for encoding.
CANONICAL_SKILLS = [
    # AI/ML
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "computer vision", "reinforcement learning",
    "neural networks", "data science", "MLOps",
    # Frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "langchain", "transformers", "hugging face",
    # Web
    "html", "css", "javascript", "typescript", "react", "angular", "vue",
    "node.js", "express", "flask", "django", "fastapi", "spring boot",
    "next.js", "graphql", "rest api",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
    "ansible", "ci/cd", "linux", "nginx",
    # Data / DB
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "elasticsearch", "apache spark", "hadoop", "kafka", "airflow",
    "power bi", "tableau", "etl", "data warehouse",
    # Languages
    "python", "java", "c++", "c#", "golang", "rust", "scala",
    "kotlin", "swift", "ruby", "php", "matlab", "r programming",
    "bash", "perl",
    # Engineering
    "autocad", "solidworks", "catia", "plc", "scada",
    "hydraulic systems", "lean manufacturing", "six sigma",
    "calibration", "test engineering", "electronics",
    # Soft Skills
    "project management", "leadership", "team management",
    "agile", "scrum", "kanban", "communication",
    "problem solving", "critical thinking", "time management",
]

_canonical_embeddings = None


def _get_canonical_embeddings():
    """Encode canonical skills once and cache."""
    global _canonical_embeddings
    if _canonical_embeddings is not None:
        return _canonical_embeddings
    model = _get_sentence_model()
    if model is None:
        return None
    _canonical_embeddings = model.encode(CANONICAL_SKILLS, normalize_embeddings=True)
    return _canonical_embeddings


# ── spaCy NER extraction ─────────────────────────────────────

def _extract_entities_spacy(text: str) -> list[str]:
    """Extract technology/skill-like entities using spaCy NER."""
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    candidates = []
    # Labels that may carry skill info
    useful_labels = {"ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "LANGUAGE", "FAC"}

    for ent in doc.ents:
        if ent.label_ in useful_labels and 2 <= len(ent.text) <= 40:
            candidates.append(ent.text.strip())

    # Also extract noun chunks that look "techy"
    tech_indicators = {
        "system", "platform", "framework", "tool", "engine", "library",
        "database", "server", "service", "protocol", "api", "sdk",
        "architecture", "model", "network", "pipeline", "deployment",
    }
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower()
        if any(ind in chunk_lower for ind in tech_indicators) and len(chunk.text) > 3:
            candidates.append(chunk.text.strip())

    return list(set(candidates))


# ── Semantic similarity matching ──────────────────────────────

def _semantic_match_skills(
    candidate_phrases: list[str],
    threshold: float = 0.55,
) -> list[dict]:
    """
    Compare candidate phrases against canonical skills via cosine similarity.

    Returns list of {"phrase": str, "matched_skill": str, "score": float}
    for matches above threshold.
    """
    model = _get_sentence_model()
    canonical_emb = _get_canonical_embeddings()
    if model is None or canonical_emb is None:
        return []

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        if not candidate_phrases:
            return []

        cand_emb = model.encode(candidate_phrases, normalize_embeddings=True)
        sim_matrix = cos_sim(cand_emb, canonical_emb)  # shape (n_cand, n_canonical)

        matches = []
        for i, phrase in enumerate(candidate_phrases):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            if best_score >= threshold:
                matches.append({
                    "phrase": phrase,
                    "matched_skill": CANONICAL_SKILLS[best_idx],
                    "score": round(best_score, 3),
                })
        return matches
    except Exception as e:
        print(f"Semantic matching error: {e}", file=sys.stderr)
        return []


# ── Synonym expansion (deterministic, fast) ───────────────────

_SYNONYM_MAP = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ai": "artificial intelligence",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "js": "javascript",
    "ts": "typescript",
    "pg": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    "r": "r programming",
    "cpp": "c++",
    "csharp": "c#",
    "sci-kit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "hf": "hugging face",
    "huggingface": "hugging face",
    "gke": "kubernetes",
    "eks": "kubernetes",
    "aks": "kubernetes",
    "ec2": "aws",
    "s3": "aws",
    "lambda": "aws",
    "sagemaker": "aws",
    "vue.js": "vue",
    "react.js": "react",
    "angular.js": "angular",
    "node": "node.js",
    "mssql": "sql",
    "t-sql": "sql",
    "plsql": "sql",
}


def _expand_synonyms(text_lower: str, existing_skills_lower: Set[str]) -> list[str]:
    """Find synonym keywords in text and return canonical skill names."""
    new_skills = []
    for abbr, canonical in _SYNONYM_MAP.items():
        if canonical.lower() in existing_skills_lower:
            continue  # Already found via regex
        # Use word boundary to avoid false positives
        pattern = r'\b' + re.escape(abbr) + r'\b'
        if re.search(pattern, text_lower):
            new_skills.append(canonical.title())
            existing_skills_lower.add(canonical.lower())
    return new_skills


# ── Public API ────────────────────────────────────────────────

@tool
def semantic_skill_enhancer(
    cv_text: str,
    regex_skills: Optional[List[str]] = None,
) -> dict:
    """
    Enhance skill extraction with semantic matching and NER.

    Takes the CV text and optionally the skills already found by regex,
    then discovers additional skills via:
        1. Synonym expansion (deterministic)
        2. spaCy NER → embedding match
        3. CV phrase chunks → embedding match

    Args:
        cv_text: Full text of the CV.
        regex_skills: Skills already found by regex (to avoid duplicates).

    Returns:
        dict with:
            - new_skills: list of newly discovered skills
            - semantic_matches: list of {phrase, matched_skill, score}
            - synonym_expansions: list of skills found via synonym mapping
            - ner_entities: raw entities found by spaCy
    """
    if not cv_text or len(cv_text.strip()) < 20:
        return {
            "new_skills": [],
            "semantic_matches": [],
            "synonym_expansions": [],
            "ner_entities": [],
        }

    existing_lower: Set[str] = set()
    if regex_skills:
        existing_lower = {s.lower() for s in regex_skills}

    all_new: list[str] = []

    # ── 1. Synonym expansion ──────────────────────────────
    synonym_found = _expand_synonyms(cv_text.lower(), existing_lower)
    all_new.extend(synonym_found)

    # ── 2. spaCy NER entities ─────────────────────────────
    ner_entities = _extract_entities_spacy(cv_text)

    # ── 3. Semantic matching on NER entities ──────────────
    # Also extract short phrases from CV (bigrams/trigrams from sentences)
    cv_phrases = _extract_cv_phrases(cv_text)
    all_candidates = list(set(ner_entities + cv_phrases))

    # Filter out candidates already matched
    all_candidates = [c for c in all_candidates if c.lower() not in existing_lower]

    semantic_matches = _semantic_match_skills(all_candidates, threshold=0.55)

    # Add matched skills
    for match in semantic_matches:
        skill_name = match["matched_skill"].title()
        if skill_name.lower() not in existing_lower:
            all_new.append(skill_name)
            existing_lower.add(skill_name.lower())

    return {
        "new_skills": all_new,
        "semantic_matches": semantic_matches,
        "synonym_expansions": synonym_found,
        "ner_entities": ner_entities[:20],  # Limit for readability
    }


def _extract_cv_phrases(text: str, max_phrases: int = 60) -> list[str]:
    """Extract meaningful short phrases from CV text for semantic comparison."""
    # Split into sentences, then extract 2-4 word chunks
    sentences = re.split(r'[.\n;]', text)
    phrases = []

    for sent in sentences:
        words = sent.strip().split()
        if len(words) < 2:
            continue
        # Bigrams and trigrams
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                # Filter: must have some alpha chars, not too short
                if len(phrase) > 4 and re.search(r'[a-zA-Z]{2,}', phrase):
                    phrases.append(phrase.strip())

    # Deduplicate and limit
    seen = set()
    unique = []
    for p in phrases:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:max_phrases]


def semantic_enhance_skills(cv_text: str, regex_skills: list[str]) -> list[str]:
    """
    Convenience function: returns a merged skill list (regex + semantic).
    Non-tool version for direct Python calls.
    """
    result = semantic_skill_enhancer.invoke({
        "cv_text": cv_text,
        "regex_skills": regex_skills,
    })
    new_skills = result.get("new_skills", [])
    return regex_skills + new_skills
