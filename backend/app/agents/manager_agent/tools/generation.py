"""
Content Generation Tools

This module contains tools for generating and validating job offers,
and checking market salary ranges.

Group: Hiring Manager Agent (Group 2)

Tools:
- job_offer_generator: Generate personalized job offers from templates
- offer_validator_tool: Sanity check for placeholders
- market_salary_check: Validate salary against market ranges
"""

import re
from typing import Optional
from langchain_core.tools import tool


# ── Built-in salary data (mock but functional) ──────────────
SALARY_DATABASE = {
    "software engineer":       {"min": 45000, "max": 95000,  "median": 65000},
    "senior software engineer": {"min": 70000, "max": 130000, "median": 95000},
    "data scientist":          {"min": 50000, "max": 110000, "median": 75000},
    "senior data scientist":   {"min": 80000, "max": 140000, "median": 105000},
    "machine learning engineer": {"min": 60000, "max": 130000, "median": 90000},
    "ai engineer":             {"min": 65000, "max": 140000, "median": 95000},
    "senior ai engineer":      {"min": 90000, "max": 160000, "median": 120000},
    "devops engineer":         {"min": 50000, "max": 110000, "median": 75000},
    "frontend developer":      {"min": 40000, "max": 90000,  "median": 60000},
    "backend developer":       {"min": 45000, "max": 100000, "median": 68000},
    "fullstack developer":     {"min": 45000, "max": 105000, "median": 70000},
    "product manager":         {"min": 55000, "max": 120000, "median": 82000},
    "project manager":         {"min": 45000, "max": 100000, "median": 68000},
    "data analyst":            {"min": 35000, "max": 75000,  "median": 52000},
    "ux designer":             {"min": 40000, "max": 90000,  "median": 60000},
    "qa engineer":             {"min": 38000, "max": 80000,  "median": 55000},
    "intern":                  {"min": 8000,  "max": 20000,  "median": 14000},
}

# ── Default offer template ──────────────────────────────────
DEFAULT_OFFER_TEMPLATE = """
# Job Offer Letter

**Date:** {date}

**Dear {candidate_name},**

We are pleased to extend an offer of employment for the position of **{job_title}** at **{company_name}**.

## Position Details

- **Job Title:** {job_title}
- **Department:** {department}
- **Location:** {location}
- **Start Date:** {start_date}
- **Contract Type:** {contract_type}

## Compensation

- **Annual Salary:** {salary} {currency}
- **Benefits Package:** {benefits}

## Key Responsibilities

{responsibilities}

## What We Offer

{company_benefits}

## Next Steps

Please confirm your acceptance of this offer by **{response_deadline}**.

We look forward to welcoming you to the team!

Best regards,
**{hiring_manager}**
{company_name}
"""


class SafeDict(dict):
    """Dict subclass that returns the key as placeholder for missing keys."""
    def __missing__(self, key):
        return f"[{key.upper()}]"


@tool
def job_offer_generator(
    template: str,
    candidate_data: dict,
    job_data: dict
) -> dict:
    """
    Generate a personalized job offer by filling a template with candidate and job data.

    Args:
        template: Template string with {variable} placeholders. If empty, uses default template.
        candidate_data: Dict with candidate info (name, skills, experience, etc.).
        job_data: Dict with job info (title, salary, location, company, etc.).

    Returns:
        Dictionary with:
        - success: bool
        - offer_text: The generated offer text
        - unfilled_count: Number of placeholders that could not be filled
    """
    if not template or not template.strip():
        template = DEFAULT_OFFER_TEMPLATE

    # Merge candidate and job data into a single context
    context = SafeDict()

    # Candidate fields
    context["candidate_name"] = candidate_data.get("name", candidate_data.get("candidate_name", "[CANDIDATE NAME]"))
    context["candidate_email"] = candidate_data.get("email", "[CANDIDATE EMAIL]")
    skills = candidate_data.get("skills", [])
    context["candidate_skills"] = ", ".join(skills) if isinstance(skills, list) else str(skills)
    context["experience_years"] = str(candidate_data.get("experience_years", "[EXPERIENCE]"))

    # Job fields
    context["job_title"] = job_data.get("title", job_data.get("job_title", "[JOB TITLE]"))
    context["company_name"] = job_data.get("company", job_data.get("company_name", "ATIA Club ESB"))
    context["department"] = job_data.get("department", "Engineering")
    context["location"] = job_data.get("location", "[LOCATION]")
    context["salary"] = str(job_data.get("salary", "[SALARY]"))
    context["currency"] = job_data.get("currency", "USD")
    context["contract_type"] = job_data.get("contract_type", "Full-time")
    context["start_date"] = job_data.get("start_date", "[START DATE]")
    context["response_deadline"] = job_data.get("response_deadline", "[RESPONSE DEADLINE]")
    context["hiring_manager"] = job_data.get("hiring_manager", "[HIRING MANAGER]")
    context["date"] = job_data.get("date", "[DATE]")
    context["benefits"] = job_data.get("benefits", "Standard company benefits package")
    context["company_benefits"] = job_data.get("company_benefits", "- Competitive salary\n- Health insurance\n- Remote work flexibility\n- Professional development budget")
    context["responsibilities"] = job_data.get("responsibilities", "- As discussed during the interview process")

    try:
        offer_text = template.format_map(context)

        # Count remaining unfilled placeholders
        unfilled = re.findall(r"\[([A-Z_ ]+)\]", offer_text)

        return {
            "success": True,
            "offer_text": offer_text.strip(),
            "unfilled_count": len(unfilled),
            "unfilled_placeholders": unfilled,
        }
    except Exception as e:
        return {
            "success": False,
            "offer_text": "",
            "error": f"Template rendering failed: {str(e)}",
        }


@tool
def offer_validator_tool(generated_text: str) -> dict:
    """
    Sanity check for generated offer text.

    Checks that placeholders like [INSERT SALARY] or [CANDIDATE NAME]
    have been properly replaced before sending.

    Args:
        generated_text: The generated offer/email text to validate.

    Returns:
        A dictionary containing:
        - valid: Boolean indicating if text is ready to send
        - unfilled_placeholders: List of placeholders still present
        - warnings: List of potential issues
        - suggestions: Recommended fixes
    """
    # Safety check for empty or too short text
    if not generated_text or len(generated_text.strip()) < 50:
        return {
            "valid": False,
            "unfilled_placeholders": [],
            "warnings": ["Offer text is empty or too short"],
            "suggestions": ["Provide a complete job offer text"]
        }

    # Detect placeholders like [INSERT SALARY], [CANDIDATE NAME], etc.
    placeholders = re.findall(r"\[.*?\]", generated_text)

    # Check for presence of critical fields
    critical_fields = ["salary", "job title", "location", "contract"]
    warnings = []
    text_lower = generated_text.lower()
    for field in critical_fields:
        if field not in text_lower:
            warnings.append(f"Missing critical field: {field}")

    # Build suggestions for placeholders
    suggestions = [f"Replace placeholder {ph}" for ph in placeholders]

    # Determine overall validity
    is_valid = len(placeholders) == 0 and len(warnings) == 0

    return {
        "valid": is_valid,
        "unfilled_placeholders": placeholders,
        "warnings": warnings,
        "suggestions": suggestions
    }


def get_salary_range(role: str, location: Optional[str] = None) -> dict:
    """
    Get market salary range for a role.

    Args:
        role: Job role/title.
        location: Optional location (for future regional adjustments).

    Returns:
        Salary range dictionary or None if role not found.
    """
    role_lower = role.lower().strip()

    # Exact match
    if role_lower in SALARY_DATABASE:
        return SALARY_DATABASE[role_lower].copy()

    # Fuzzy match: check if any key is contained in the role
    for key, data in SALARY_DATABASE.items():
        if key in role_lower or role_lower in key:
            return data.copy()

    return None


@tool
def market_salary_check(
    role: str,
    offered_salary: float,
    location: Optional[str] = None
) -> dict:
    """
    Check if offered salary is within market range.

    Uses a dictionary of salary ranges to flag if the
    generated offer salary is too low (or too high).

    Args:
        role: Job role/title to check.
        offered_salary: The salary amount in the offer.
        location: Optional location for regional adjustment.

    Returns:
        A dictionary containing:
        - within_range: Boolean indicating if salary is acceptable
        - market_min: Minimum market salary for role
        - market_max: Maximum market salary for role
        - market_median: Median market salary
        - deviation_percent: How far from median (+ or -)
        - flag: 'low', 'high', or 'ok'
        - recommendation: Suggested action if out of range
    """
    salary_range = get_salary_range(role, location)

    if salary_range is None:
        return {
            "within_range": None,
            "market_min": None,
            "market_max": None,
            "market_median": None,
            "deviation_percent": None,
            "flag": "unknown",
            "recommendation": (
                f"No salary data found for role '{role}'. "
                f"Available roles: {', '.join(sorted(SALARY_DATABASE.keys()))}"
            )
        }

    market_min = salary_range["min"]
    market_max = salary_range["max"]
    market_median = salary_range["median"]

    deviation = ((offered_salary - market_median) / market_median) * 100

    if offered_salary < market_min:
        flag = "low"
        within_range = False
        recommendation = (
            f"⚠️ Offered salary ({offered_salary:,.0f}) is BELOW market minimum ({market_min:,.0f}). "
            f"Consider raising to at least {market_min:,.0f} to attract qualified candidates."
        )
    elif offered_salary > market_max:
        flag = "high"
        within_range = False
        recommendation = (
            f"💰 Offered salary ({offered_salary:,.0f}) is ABOVE market maximum ({market_max:,.0f}). "
            f"This is generous but may indicate budget inefficiency."
        )
    else:
        flag = "ok"
        within_range = True
        recommendation = (
            f"✅ Offered salary ({offered_salary:,.0f}) is within market range "
            f"({market_min:,.0f} – {market_max:,.0f}). Median: {market_median:,.0f}."
        )

    return {
        "within_range": within_range,
        "market_min": market_min,
        "market_max": market_max,
        "market_median": market_median,
        "deviation_percent": round(deviation, 1),
        "flag": flag,
        "recommendation": recommendation,
    }


# ── Non-tool helpers (kept for backwards compatibility) ─────

def validate_offer(content):
    """Non-tool version of offer_validator_tool."""
    return offer_validator_tool.invoke({"generated_text": content})
