from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
from app.agents.recruiter_agent.tools.parsers import cv_parser_tool
from app.agents.recruiter_agent.tools.extraction import skill_extractor_tool, candidate_summarizer
from app.agents.recruiter_agent.tools.anonymizer_tool import anonymizer_tool
from app.agents.recruiter_agent.tools.semantic_extractor import semantic_skill_enhancer
from app.agents.recruiter_agent.tools.llm_ranker import rank_single_candidate

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# In-memory cache of parsed CVs for ranking (per-session simple store)
_parsed_cv_cache: dict[str, dict] = {}


@router.post("/analyze")
async def analyze_cv(file: UploadFile = File(...)):
    """Parse a CV, extract skills, generate summary, and cache for ranking."""
    try:
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        # 1. Parse CV (now with OCR fallback for scanned PDFs)
        with open(file_location, "rb") as f:
            parse_result = cv_parser_tool.invoke({"file_obj": f})

        if parse_result.get("error"):
            raise HTTPException(status_code=400, detail=parse_result["error"])

        text = parse_result["text"]

        # 2. Anonymize PII before processing (privacy-first)
        anon_result = anonymizer_tool.invoke({"cv_text": text})
        anonymized_text = anon_result.get("anonymized_text", text)
        redactions = anon_result.get("redactions", {})

        # 3. Extract Skills (regex-based, deterministic)
        skills_data = skill_extractor_tool.invoke({"cv_text": text})

        # 4. Semantic enhancement (embeddings + synonym expansion + NER)
        try:
            regex_skills = skills_data.get("skills", []) if isinstance(skills_data, dict) else []
            semantic_result = semantic_skill_enhancer.invoke({
                "cv_text": text,
                "regex_skills": regex_skills,
            })
            new_skills = semantic_result.get("new_skills", [])
            if new_skills and isinstance(skills_data, dict):
                # Merge: add semantic skills to the main list
                existing_lower = {s.lower() for s in skills_data["skills"]}
                for ns in new_skills:
                    if ns.lower() not in existing_lower:
                        skills_data["skills"].append(ns)
                        existing_lower.add(ns.lower())
                skills_data["semantic_matches"] = semantic_result.get("semantic_matches", [])
                skills_data["synonym_expansions"] = semantic_result.get("synonym_expansions", [])
        except Exception as e:
            print(f"Semantic enhancement skipped: {e}")

        # 5. Generate Summary
        summary = candidate_summarizer.invoke({
            "cv_text": text,
            "extracted_skills": skills_data
        })

        # 6. Cache for later ranking (store both original and anonymized)
        _parsed_cv_cache[file.filename] = {
            "text": text,
            "anonymized_text": anonymized_text,
            "skills_data": skills_data,
            "summary": summary,
        }

        # Build a frontend-friendly response
        skills_list = skills_data.get("skills", []) if isinstance(skills_data, dict) else []
        experience_years = skills_data.get("experience_years", 0) if isinstance(skills_data, dict) else 0
        candidate_name = skills_data.get("candidate_name", None) if isinstance(skills_data, dict) else None
        job_title = skills_data.get("job_title", None) if isinstance(skills_data, dict) else None

        return {
            "filename": file.filename,
            "text": text,
            "skills_data": skills_data,
            "skills": skills_list,
            "total_experience": experience_years,
            "candidate_name": candidate_name,
            "job_title": job_title,
            "summary": summary,
            "pages": parse_result.get("pages", 0),
            "word_count": parse_result.get("word_count", 0),
            "extraction_method": parse_result.get("extraction_method", "unknown"),
            "ocr_confidence": parse_result.get("ocr_confidence"),
            "pii_redactions": redactions if redactions else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RankRequest(BaseModel):
    job_description: str
    filenames: Optional[List[str]] = None  # if None, rank all cached CVs


@router.post("/rank")
async def rank_candidates(request: RankRequest):
    """
    Rank previously analyzed CVs against a job description.
    Uses llm_rank_candidates (60% embeddings + 40% skill overlap via Mistral).
    """
    if not _parsed_cv_cache:
        raise HTTPException(
            status_code=400,
            detail="No CVs analyzed yet. Please upload and analyze CVs first via /candidates/analyze."
        )

    # Select which CVs to rank
    filenames = request.filenames or list(_parsed_cv_cache.keys())
    missing = [f for f in filenames if f not in _parsed_cv_cache]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"CVs not found in cache: {missing}. Analyze them first."
        )

    # Rank each candidate
    rankings = []
    for fname in filenames:
        cached = _parsed_cv_cache[fname]
        result = rank_single_candidate(
            job_description=request.job_description,
            cv_text=cached["text"],
            candidate_name=fname,
            use_llm=True,
        )
        result["summary"] = cached.get("summary", "")
        rankings.append(result)

    # Sort by score descending
    rankings.sort(key=lambda x: x["score"], reverse=True)

    return {
        "job_description_preview": request.job_description[:200] + "...",
        "total_candidates": len(rankings),
        "rankings": rankings,
    }


@router.get("/cached")
async def list_cached_cvs():
    """List all CVs currently in the analysis cache (available for ranking)."""
    return {
        "cached_cvs": [
            {
                "filename": fname,
                "skills_count": len(data.get("skills_data", {}).get("skills", [])),
                "summary_preview": str(data.get("summary", ""))[:100],
            }
            for fname, data in _parsed_cv_cache.items()
        ],
        "total": len(_parsed_cv_cache),
    }


@router.delete("/cached/{filename}")
async def delete_cached_cv(filename: str):
    """Delete a CV from the analysis cache and remove the uploaded file."""
    if filename not in _parsed_cv_cache:
        raise HTTPException(status_code=404, detail=f"CV '{filename}' not found in cache.")

    del _parsed_cv_cache[filename]

    # Also remove the uploaded file from disk
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"deleted": filename, "remaining": len(_parsed_cv_cache)}


# Helper function for agent access
def get_cv_cache():
    """Returns the CV cache for agent access."""
    return _parsed_cv_cache
