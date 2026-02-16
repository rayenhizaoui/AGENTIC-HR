"""
Skill Extraction Tools

This module contains tools for extracting skills, generating summaries,
and normalizing experience data from CV text.
"""

from typing import Optional, List, Tuple
from langchain_core.tools import tool
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# JSON Schema for structured skill extraction output
SKILL_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of technical and soft skills"
        },
        "experience_years": {
            "type": "integer",
            "description": "Total years of professional experience"
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": "integer"}
                }
            }
        },
        "certifications": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Professional certifications"
        },
        "projects_count": {
            "type": "integer",
            "description": "Number of projects identified"
        }
    },
    "required": ["skills", "experience_years", "education", "certifications"]
}


@tool
def skill_extractor_tool(cv_text: str) -> dict:
    """
    Extract structured data from CV text using word-boundary-aware keyword matching.
    Falls back to LLM if configured.
    """
    cv_text_lower = cv_text.lower()
    
    # ── Skill categories with word-boundary-safe keywords ──
    # Short keywords (<=3 chars) need exact word boundary matching
    # Longer keywords can use simple containment but we'll use boundary for all
    skill_categories = {
        "AI/ML": ["machine learning", "deep learning", "artificial intelligence",
                  "natural language processing", "computer vision",
                  "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
                  "langchain", "transformers", "prompt engineering", "data science",
                  "neural network", "reinforcement learning", "opencv"],
        "Web/Fullstack": ["html", "css", "javascript", "typescript", "react", "angular", "vue",
                          "node.js", "nodejs", "express", "flask", "django", "fastapi",
                          "spring boot", "next.js", "nuxt", "svelte", "graphql",
                          "rest api", "restful"],
        "Cloud/DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
                         "ci/cd", "terraform", "ansible", "grafana", "prometheus",
                         "linux", "nginx", "apache"],
        "Data/DB": ["sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
                    "elasticsearch", "cassandra", "oracle", "sqlite", "bigquery",
                    "snowflake", "apache spark", "hadoop", "kafka", "airflow",
                    "power bi", "tableau", "excel", "dax", "etl"],
        "Languages": ["python", "java", "c\\+\\+", "c#", "golang", "rust", "scala",
                      "kotlin", "swift", "ruby", "php", "matlab", "labview",
                      "dasylab", "vba", "perl", "bash"],
        "Engineering": ["autocad", "solidworks", "catia", "plc", "scada",
                       "hydraulic", "pneumatic", "lean", "six sigma", "5s",
                       "iso 9001", "fea", "cfd", "test stand", "calibration",
                       "fabrication", "electronics"],
        "Soft Skills": ["project management", "leadership", "team lead",
                       "agile", "scrum", "kanban", "communication"]
    }

    found_skills = []
    found_categories = {}

    for cat, keywords in skill_categories.items():
        cat_skills = []
        for skill in keywords:
            # Use word boundary regex for ALL keywords to avoid substring false positives
            # e.g. "r" should NOT match "career", "ai" should NOT match "maintain"
            pattern = r'\b' + skill + r'\b'
            if re.search(pattern, cv_text_lower):
                # Store the clean display name  
                display = skill.replace("\\+", "+").title()
                cat_skills.append(display)
                found_skills.append(display)
        if cat_skills:
            found_categories[cat] = cat_skills

    # Also scan for a dedicated "Skills" section in the CV
    skills_section = _extract_skills_section(cv_text)
    if skills_section:
        # Add any skills from the dedicated section that we haven't found yet
        section_items = [s.strip() for s in re.split(r'[,;|•·]', skills_section) if s.strip() and len(s.strip()) > 1]
        for item in section_items:
            clean = item.strip().title()
            if clean and clean not in found_skills and len(clean) > 2:
                found_skills.append(clean)

    # Deduplicate preserving order
    seen = set()
    unique_skills = []
    for s in found_skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique_skills.append(s)
    found_skills = unique_skills
    
    # ── Advanced Experience Extraction with Overlap Detection ──
    years = 0
    date_ranges = _extract_date_ranges(cv_text)

    if date_ranges:
        # Merge overlapping ranges, then sum non-overlapping durations
        years = _compute_experience_from_ranges(date_ranges)

    # Fallback 1: Look for explicit "X years experience"
    if not years:
        match = re.search(r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)', cv_text_lower)
        if match:
            years = int(match.group(1))

    # Fallback 2: Min/max year heuristic
    if not years:
        year_matches = re.findall(r'\b(20\d{2})\b', cv_text)
        if len(year_matches) >= 2:
            dates = [int(y) for y in year_matches]
            diff = max(dates) - min(dates)
            if 0 < diff < 40:
                years = diff

    # Fallback 3: Intern heuristic
    is_intern = "intern" in cv_text_lower
    if years == 0 and is_intern:
        intern_count = cv_text_lower.count("intern")
        if intern_count >= 2:
            years = 1  # Multiple internships ≈ 1 year
    
    
    # Improved Education Extraction — require degree context, not just "engineer" anywhere
    education = []
    
    # Degree patterns: (regex, degree_label)
    degree_patterns = [
        (r'\b(associates?|associate\'?s?)\b.*?\b(degree|science|arts|applied)\b', "Associate's Degree"),
        (r'\b(bachelor\'?s?|baccalaureate|licence|b\.?s\.?|b\.?a\.?)\b.*?\b(degree|science|arts|engineering|in)\b', "Bachelor's Degree"),
        (r'\b(engineering\s+degree|diplome\s+d\'?ingenieur|ingenieur|engineer\'?s?\s+degree)\b', "Engineering Degree"),
        (r'\b(master\'?s?|m\.?s\.?|m\.?a\.?)\b.*?\b(degree|science|arts|in|of)\b', "Master's Degree"),
        (r'\b(ph\.?d\.?|doctorate|doctorat)\b', "PhD/Doctorate"),
    ]

    for pattern, deg_label in degree_patterns:
        for m in re.finditer(pattern, cv_text_lower):
            field = _extract_education_field(cv_text_lower, m.start())
            institution = _extract_institution(cv_text, m.start())
            year = _extract_education_year(cv_text, m.start())
            entry = {
                "degree": deg_label,
                "field": field,
                "institution": institution,
            }
            if year:
                entry["year"] = year
            education.append(entry)

    # Deduplicate education entries
    unique_edu = []
    seen_degrees = set()
    for edu in education:
        key = f"{edu['degree']}|{edu.get('field', '')}"
        if key not in seen_degrees:
            unique_edu.append(edu)
            seen_degrees.add(key)
    
    # Extract Project Count
    project_count = len(re.findall(r'\bproject\b', cv_text_lower))

    # Extract candidate name (usually the first substantial line)
    candidate_name = _extract_candidate_name(cv_text)

    # Extract most recent job title
    job_title = _extract_job_title(cv_text)

    return {
        "skills": found_skills if found_skills else ["No specific skills detected"],
        "skill_categories": found_categories,
        "experience_years": years,
        "education": unique_edu,
        "certifications": [],
        "projects_count": project_count,
        "candidate_name": candidate_name,
        "job_title": job_title,
        "note": "Extracted using word-boundary keyword matching.",
        "date_ranges": [(str(s), str(e)) for s, e in date_ranges] if date_ranges else [],
    }


# ── Date-Range Extraction & Overlap Merging ─────────────────

_MONTH_MAP = {
    "jan": 1, "january": 1, "janvier": 1,
    "feb": 2, "february": 2, "février": 2, "fevrier": 2,
    "mar": 3, "march": 3, "mars": 3,
    "apr": 4, "april": 4, "avril": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juin": 6,
    "jul": 7, "july": 7, "juillet": 7,
    "aug": 8, "august": 8, "août": 8, "aout": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "décembre": 12, "decembre": 12,
}


def _parse_date_token(token: str) -> Optional[datetime]:
    """Parse a date token like 'Jan 2020', '2020', 'Present', 'Current'."""
    token = token.strip().lower()
    now = datetime.now()

    if token in ("present", "current", "aujourd'hui", "actuel", "now", "ce jour"):
        return now

    # "Month Year" pattern
    m = re.match(r'([a-zéûô]+)\s*(\d{4})', token)
    if m:
        month_str, year_str = m.group(1), m.group(2)
        month = _MONTH_MAP.get(month_str)
        if month:
            return datetime(int(year_str), month, 1)

    # Just a year
    m = re.match(r'^(\d{4})$', token)
    if m:
        year = int(m.group(1))
        if 1970 <= year <= now.year + 1:
            return datetime(year, 1, 1)

    return None


def _extract_date_ranges(cv_text: str) -> List[Tuple[datetime, datetime]]:
    """
    Extract employment date ranges from CV text.

    Looks for patterns like:
      - "Jan 2020 - Dec 2022"
      - "2018 to Present"
      - "Mar 2015 – Jun 2018"
      - "Oct 2016 to Current"
    """
    ranges = []

    # Pattern: Month Year - Month Year  OR  Year - Year
    pattern = (
        r'(?:([A-Za-zéûô]+)\s+)?'     # optional month
        r'(\d{4})'                       # start year
        r'\s*(?:[-–—]|to|à)\s*'         # separator
        r'(?:'
            r'(?:([A-Za-zéûô]+)\s+)?'   # optional end month
            r'(\d{4})'                   # end year
        r'|'
            r'(present|current|aujourd\'?hui|actuel|now|ce\s+jour)'  # or "Present"
        r')'
    )

    for m in re.finditer(pattern, cv_text, re.IGNORECASE):
        start_month = m.group(1) or ""
        start_year = m.group(2)
        end_month = m.group(3) or ""
        end_year = m.group(4)
        end_present = m.group(5)

        start_token = f"{start_month} {start_year}".strip()
        if end_present:
            end_token = "present"
        else:
            end_token = f"{end_month} {end_year}".strip()

        start_dt = _parse_date_token(start_token)
        end_dt = _parse_date_token(end_token)

        if start_dt and end_dt and start_dt <= end_dt:
            ranges.append((start_dt, end_dt))

    return ranges


def _merge_overlapping_ranges(ranges: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """Merge overlapping or adjacent date ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping — extend
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _compute_experience_from_ranges(ranges: List[Tuple[datetime, datetime]]) -> int:
    """Merge overlapping ranges, then compute total non-overlapping years."""
    merged = _merge_overlapping_ranges(ranges)
    total_days = sum((end - start).days for start, end in merged)
    return max(0, round(total_days / 365.25))


def _extract_education_year(cv_text: str, pos: int) -> Optional[int]:
    """Extract graduation year near a degree mention (same line only)."""
    # Get the line containing the degree
    line_start = cv_text.rfind('\n', 0, pos)
    line_start = line_start + 1 if line_start != -1 else 0
    line_end = cv_text.find('\n', pos)
    if line_end == -1:
        line_end = len(cv_text)
    line = cv_text[line_start:line_end]
    
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', line)
    if years:
        # Return the latest year on this line (graduation year)
        return max(int(y) for y in years)
    return None


def _extract_skills_section(cv_text: str) -> Optional[str]:
    """Extract text from a dedicated 'Skills' section in the CV."""
    # Look for a section header like "Skills", "Technical Skills", "Compétences"
    pattern = r'(?:^|\n)\s*(?:skills|technical\s+skills|comp[eé]tences|technologies|outils)\s*[:\n](.+?)(?:\n\s*(?:education|experience|formation|training|certification|references|projects|interests|hobbies)\b|\Z)'
    match = re.search(pattern, cv_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_education_field(cv_text_lower: str, pos: int) -> str:
    """Extract the field of study near a degree mention."""
    # Look in the ~200 chars around the degree mention
    context = cv_text_lower[max(0, pos):min(len(cv_text_lower), pos + 200)]
    
    field_patterns = [
        (r'electronics?\s+engineering', "Electronics Engineering"),
        (r'electrical\s+engineering', "Electrical Engineering"),
        (r'mechanical\s+engineering', "Mechanical Engineering"),
        (r'computer\s+science', "Computer Science"),
        (r'software\s+engineering', "Software Engineering"),
        (r'data\s+science', "Data Science"),
        (r'information\s+(?:technology|systems)', "Information Technology"),
        (r'business\s+administration', "Business Administration"),
        (r'munitions?\s+systems?\s+technology', "Munitions Systems Technology"),
        (r'applied\s+science', "Applied Science"),
        (r'mathematics|math', "Mathematics"),
        (r'physics', "Physics"),
        (r'artificial\s+intelligence', "Artificial Intelligence"),
        (r'cyber\s*security', "Cybersecurity"),
        (r'telecommunications?', "Telecommunications"),
    ]
    
    for pattern, field in field_patterns:
        if re.search(pattern, context):
            return field
    return "General Studies"


def _extract_institution(cv_text: str, pos: int) -> str:
    """Try to extract institution name near a degree mention."""
    # Get the line containing the degree mention
    line_start = cv_text.rfind('\n', 0, pos)
    line_start = line_start + 1 if line_start != -1 else 0
    line_end = cv_text.find('\n', pos)
    if line_end == -1:
        line_end = len(cv_text)
    # Extend to also look at the next line
    next_line_end = cv_text.find('\n', line_end + 1)
    if next_line_end == -1:
        next_line_end = len(cv_text)
    context = cv_text[line_start:next_line_end]
    
    # Known short-name institutions  
    known_short = [
        "MIT", "INSAT", "ENIT", "ENSI", "ESPRIT", "ENIS", "ISET",
        "ISIMS", "FST", "ISG", "IHEC", "ENP", "EPFL", "ETH", "UCLA",
        "NYU", "USC", "CMU", "UC Berkeley", "CalTech",
    ]
    for short in known_short:
        if short in context:
            return short

    # Look for "University", "College", "Institute" etc.
    inst_match = re.search(
        r'\b([A-Z][A-Za-zÀ-ÿ\s&\'-]{2,50}(?:University|College|Institute|School|Polytechnic|'
        r'Université|Ecole|École|Faculté))\b',
        context
    )
    if inst_match:
        inst = re.sub(r'^\d{4}\s*', '', inst_match.group(1).strip())
        return inst.strip()
    
    return "Institution (see CV)"


def _extract_candidate_name(cv_text: str) -> Optional[str]:
    """Extract candidate name — typically the first non-empty line of a CV."""
    lines = cv_text.strip().split('\n')
    for line in lines[:5]:
        clean = line.strip()
        # Skip empty lines, very short lines, or lines that look like headers
        if not clean or len(clean) < 2:
            continue
        # Skip lines that look like contact info or section headers
        if re.search(r'@|http|www\.|phone|email|address|resume|curriculum|cv\b', clean, re.IGNORECASE):
            continue
        # A name is typically 2-5 capitalized words, no numbers
        if re.match(r'^[A-ZÀ-Ž][a-zà-ž]+(?:\s+[A-ZÀ-Ž][a-zà-ž]+){0,4}$', clean):
            return clean
        # ALL CAPS name
        if re.match(r'^[A-ZÀ-Ž\s\-]{4,40}$', clean) and not re.search(r'\d', clean):
            return clean.title()
    return None


def _extract_job_title(cv_text: str) -> Optional[str]:
    """Extract the most prominent job title from the CV."""
    lines = cv_text.strip().split('\n')
    
    title_keywords = [
        'engineer', 'developer', 'manager', 'analyst', 'technician',
        'designer', 'architect', 'consultant', 'specialist', 'scientist',
        'administrator', 'coordinator', 'director', 'lead', 'intern',
        'ingénieur', 'développeur', 'technicien', 'chef'
    ]
    
    for line in lines[:10]:
        clean = line.strip()
        if not clean or len(clean) < 5 or len(clean) > 80:
            continue
        lower = clean.lower()
        for kw in title_keywords:
            if kw in lower:
                return clean
    return None


def extract_skills(text: str) -> dict:
    """Core skill extraction function (non-tool version)."""
    return skill_extractor_tool.invoke(text)


@tool
def candidate_summarizer(cv_text: str, extracted_skills: Optional[dict] = None) -> str:
    """
    Generate a professional executive summary from CV text and extracted data.
    
    Args:
        cv_text: The full text of the CV.
        extracted_skills: Optional dictionary from skill_extractor_tool.
    """
    if not cv_text or len(cv_text.strip()) < 30:
        return "Insufficient information available to generate a candidate summary."

    skills = extracted_skills.get("skills", []) if extracted_skills else []
    experience_years = extracted_skills.get("experience_years", 0) if extracted_skills else 0
    education = extracted_skills.get("education", []) if extracted_skills else []
    project_count = extracted_skills.get("projects_count", 0) if extracted_skills else 0
    job_title = extracted_skills.get("job_title", None) if extracted_skills else None
    candidate_name = extracted_skills.get("candidate_name", None) if extracted_skills else None
    skill_categories = extracted_skills.get("skill_categories", {}) if extracted_skills else {}

    # Determine seniority
    level = "Entry-Level"
    if experience_years >= 8:
        level = "Senior"
    elif experience_years >= 4:
        level = "Mid-Level"
    elif experience_years >= 1:
        level = "Junior"
    elif "intern" in cv_text.lower():
        level = "Intern/Trainee"

    # Build summary parts
    parts = []
    
    # Opening line
    name_str = f"**{candidate_name}**" if candidate_name else "The candidate"
    title_str = f" — *{job_title}*" if job_title else ""
    exp_str = f" with **{experience_years}+ years** of experience" if experience_years > 0 else ""
    parts.append(f"{name_str}{title_str}")
    parts.append(f"{level} professional{exp_str}.")

    # Skills by category
    if skill_categories:
        cat_parts = []
        for cat, cat_skills in skill_categories.items():
            cat_parts.append(f"**{cat}**: {', '.join(cat_skills)}")
        parts.append("\n**Key Competencies:**\n" + "\n".join(f"- {c}" for c in cat_parts))
    elif skills and skills != ["No specific skills detected"]:
        parts.append(f"\n**Skills:** {', '.join(skills[:12])}")

    # Education
    if education:
        edu_parts = []
        for edu in education:
            inst = edu.get('institution', '')
            edu_parts.append(f"- {edu['degree']} in {edu['field']}" + (f" — *{inst}*" if inst else ""))
        parts.append("\n**Education:**\n" + "\n".join(edu_parts))

    # Projects
    if project_count > 0:
        parts.append(f"\n**Projects mentioned:** ~{project_count}")

    return "\n".join(parts)


def _parse_year(value: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group()) if match else None


def _parse_month_year(value: str) -> Optional[datetime]:
    try:
        if not value: return None
        return datetime.strptime(value.strip(), "%b %Y")
    except ValueError:
        try:
            return datetime.strptime(value.strip(), "%B %Y")
        except ValueError:
            return None


def experience_normalizer(date_string: str) -> int:
    """
    Convert varied date formats to total years of experience.
    """
    if not date_string:
        return 0

    text = date_string.lower().strip()
    now = datetime.now()

    match_years = re.search(r"(\d+)\s+years?", text)
    if match_years:
        return int(match_years.group(1))

    if "present" in text or "current" in text:
        parts = re.split(r"-|to", text)
        if parts:
            start = _parse_month_year(parts[0].title()) 
            if not start:
                y = _parse_year(parts[0])
                if y: start = datetime(y, 1, 1)
            
            if start:
                return max(0, now.year - start.year)

    years = re.findall(r"\b(19|20)\d{2}\b", text)
    if len(years) >= 2:
        return abs(int(years[1]) - int(years[0]))

    parts = re.split(r"-|to", text)
    if len(parts) == 2:
        start = _parse_month_year(parts[0].title())
        end = _parse_month_year(parts[1].title())
        if start and end:
            return max(0, end.year - start.year)

    year = _parse_year(text)
    if year:
        return max(0, now.year - year)

    return 0


def aggregate_experience(date_ranges: List[str]) -> int:
    """Calculate total experience from a list of date ranges."""
    total = 0
    if not date_ranges: return 0
    for date_range in date_ranges:
        total += experience_normalizer(date_range)
    return total
