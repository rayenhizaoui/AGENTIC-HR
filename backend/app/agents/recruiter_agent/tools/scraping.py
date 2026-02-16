"""
Job Scraping Tools

This module contains tools for scraping job postings from
LinkedIn, Indeed, and other job boards.

Group: Recruiter Agent (Group 1)

Tools:
- job_scraper_tool: Scrape a job posting URL and return as Markdown
"""

import re
from typing import Optional, List
from langchain_core.tools import tool

try:
    import requests
    from bs4 import BeautifulSoup
    _HAS_SCRAPING = True
except ImportError:
    _HAS_SCRAPING = False
    print("Warning: requests or beautifulsoup4 not installed. job_scraper_tool will use fallback.")


# ── Supported job boards ────────────────────────────────────
SUPPORTED_BOARDS = {
    "linkedin": re.compile(r"https?://(www\.)?linkedin\.com/jobs", re.I),
    "indeed": re.compile(r"https?://(www\.)?indeed\.", re.I),
    "glassdoor": re.compile(r"https?://(www\.)?glassdoor\.", re.I),
    "welcometothejungle": re.compile(r"https?://(www\.)?welcometothejungle\.", re.I),
    "generic": re.compile(r"https?://", re.I),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def validate_job_url(url: str) -> dict:
    """
    Validate if a URL is a supported job posting URL.

    Args:
        url: URL to validate.

    Returns:
        Validation result with detected job board.
    """
    if not url or not url.strip():
        return {"valid": False, "board": None, "error": "Empty URL provided"}

    for board_name, pattern in SUPPORTED_BOARDS.items():
        if pattern.search(url):
            return {"valid": True, "board": board_name}

    return {"valid": False, "board": None, "error": "URL does not match any supported job board"}


def _extract_text_blocks(soup: BeautifulSoup) -> str:
    """Extract meaningful text blocks from HTML soup."""
    # Remove scripts, styles, navs, footers
    for element in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
        element.decompose()

    lines = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "span", "div"]):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) > 10:
            # Convert headings to markdown
            if tag.name in ("h1", "h2", "h3", "h4"):
                prefix = "#" * int(tag.name[1])
                lines.append(f"{prefix} {text}")
            elif tag.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

    # Deduplicate consecutive identical lines
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return "\n\n".join(deduped)


def parse_job_requirements(job_description_md: str) -> dict:
    """
    Parse job requirements from scraped Markdown.

    Args:
        job_description_md: Job description in Markdown format.

    Returns:
        Structured requirements dict with required_skills, preferred_skills, etc.
    """
    text_lower = job_description_md.lower()

    # Common tech skills to look for
    tech_skills = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
        "spring boot", ".net", "sql", "nosql", "mongodb", "postgresql",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "machine learning", "deep learning", "nlp", "computer vision",
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
        "git", "ci/cd", "jenkins", "agile", "scrum",
    ]

    found_skills = [s for s in tech_skills if s in text_lower]

    # Try to find experience requirement
    exp_match = re.search(r"(\d+)\+?\s*(?:years?|ans?)\s*(?:of\s+)?experience", text_lower)
    min_experience = int(exp_match.group(1)) if exp_match else None

    # Try to find education requirement
    education = []
    if "bachelor" in text_lower or "licence" in text_lower:
        education.append("Bachelor's Degree")
    if "master" in text_lower:
        education.append("Master's Degree")
    if "phd" in text_lower or "doctorate" in text_lower:
        education.append("PhD")

    return {
        "required_skills": found_skills,
        "preferred_skills": [],
        "min_experience_years": min_experience,
        "education": education,
    }


@tool
def job_scraper_tool(url: str) -> dict:
    """
    Scrape job posting from a URL and return as Markdown.

    The agent uses this when the user says:
    "Rank these CVs against this job link."

    Args:
        url: URL of the job posting (LinkedIn, Indeed, etc.).

    Returns:
        A dictionary containing:
        - success: Boolean indicating if scraping succeeded
        - job_description_md: Job description as Markdown
        - title: Extracted job title
        - company: Company name
        - location: Job location
        - requirements: Parsed requirements (if detectable)
        - error: Error message if scraping failed
    """
    result = {
        "success": False,
        "job_description_md": "",
        "title": "",
        "company": "",
        "location": "",
        "requirements": {},
        "error": None,
    }

    # Validate URL
    validation = validate_job_url(url)
    if not validation["valid"]:
        result["error"] = validation.get("error", "Invalid URL")
        return result

    if not _HAS_SCRAPING:
        result["error"] = (
            "Scraping libraries not installed. "
            "Run: pip install requests beautifulsoup4"
        )
        return result

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_tag = soup.find("h1") or soup.find("title")
        result["title"] = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

        # Extract main text content as Markdown
        job_md = _extract_text_blocks(soup)
        result["job_description_md"] = job_md

        # Parse structured requirements from the text
        result["requirements"] = parse_job_requirements(job_md)

        result["success"] = True

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out. The website took too long to respond."
    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to the website. Check the URL and your internet."
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP error: {e.response.status_code}"
    except Exception as e:
        result["error"] = f"Scraping failed: {str(e)}"

    return result


def scrape_job(url: str) -> dict:
    """
    Core job scraping function (non-tool version).

    Args:
        url: Job posting URL.

    Returns:
        Scraped job data dictionary.
    """
    return job_scraper_tool.invoke({"url": url})
