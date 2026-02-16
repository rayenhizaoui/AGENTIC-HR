"""
Job Fetcher Tools

Fetches job postings from RSS feeds and REST APIs.
Supports: Remote OK, We Work Remotely, Arbeitnow, The Muse, Remotive, Jobicy, Emploi.tn.

Features:
- Multi-keyword filtering with relevance scoring
- Proper company name extraction from RSS
- Synonym expansion (IA → AI, ML, etc.)
- Results sorted by relevance
- Tunisia / MENA job support via Emploi.tn + TanitJobs scraping

Tools:
- job_search_tool: Search for jobs across multiple sources
"""

import hashlib
import re
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False
    print("⚠️ feedparser not installed. RSS fetching disabled. Run: pip install feedparser")

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

from .job_cache import get_cached, set_cache


# ── RSS Feed Registry ────────────────────────────────────────
RSS_FEEDS = {
    "Remote OK": "https://remoteok.com/remote-dev-jobs.rss",
    "We Work Remotely": "https://weworkremotely.com/categories/remote-programming-jobs.rss",
}

# ── API Source Registry ──────────────────────────────────────
API_SOURCES = {
    "Arbeitnow": {
        "url": "https://www.arbeitnow.com/api/job-board-api",
        "needs_key": False,
    },
    "The Muse": {
        "url": "https://www.themuse.com/api/public/jobs",
        "needs_key": False,
    },
}

# ── Keyword Synonyms ────────────────────────────────────────
# Maps user search terms to expanded keywords for better matching
QUERY_SYNONYMS = {
    "ia": ["ai", "artificial intelligence", "machine learning", "ml", "deep learning", "data science", "nlp"],
    "ai": ["artificial intelligence", "machine learning", "ml", "deep learning", "data science", "nlp"],
    "ml": ["machine learning", "ai", "artificial intelligence", "data science"],
    "devops": ["devops", "site reliability", "sre", "infrastructure", "platform engineer", "ci/cd"],
    "frontend": ["frontend", "front-end", "front end", "react", "angular", "vue", "ui developer"],
    "backend": ["backend", "back-end", "back end", "server-side", "api developer"],
    "fullstack": ["fullstack", "full-stack", "full stack"],
    "data": ["data science", "data engineer", "data analyst", "analytics", "big data"],
    "cloud": ["cloud", "aws", "azure", "gcp", "infrastructure"],
    "mobile": ["mobile", "ios", "android", "react native", "flutter"],
    "security": ["security", "cybersecurity", "infosec", "penetration", "soc"],
}


def _job_hash(title: str, link: str) -> str:
    """Generate deduplication hash."""
    return hashlib.md5(f"{title.strip().lower()}|{link.strip().lower()}".encode()).hexdigest()


def _expand_query(query: str) -> list[str]:
    """
    Expand query into a list of search keywords including synonyms.
    e.g. "IA" → ["ia", "ai", "artificial intelligence", "machine learning", ...]
    """
    words = [w.strip().lower() for w in re.split(r'[\s,;]+', query) if w.strip()]
    expanded = set(words)
    for word in words:
        if word in QUERY_SYNONYMS:
            expanded.update(QUERY_SYNONYMS[word])
    return list(expanded)


def _relevance_score(title: str, description: str, tags: str, keywords: list[str]) -> int:
    """
    Calculate a relevance score (0-100) for a job against search keywords.
    Higher = more relevant.
    """
    try:
        if not keywords:
            return 50
            
        # Ensure inputs are strings
        t_safe = str(title).lower() if title else ""
        d_safe = str(description).lower() if description else ""
        tags_safe = str(tags).lower() if tags else ""
        
        text_lower = f"{t_safe} {d_safe} {tags_safe}"
        
        score = 0
        matches = 0

        for kw in keywords:
            if not isinstance(kw, str):
                continue
                
            kw_lower = kw.lower()
            if kw_lower in t_safe:
                score += 30  # High weight for title match
                matches += 1
            elif kw_lower in text_lower:
                score += 10  # Lower weight for description/tags match
                matches += 1

        # Bonus for multiple keyword matches
        if matches >= 3:
            score += 20
        elif matches >= 2:
            score += 10

        return min(score, 100)
    except Exception:
        return 0


def _clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    try:
        return re.sub(r'<[^>]+>', ' ', str(text)).strip()
    except Exception:
        return str(text)


# ═══════════════════════════════════════════════════════════════
#  RSS Fetching
# ═══════════════════════════════════════════════════════════════

def fetch_rss_jobs(feed_url: str, source_name: str, keywords: list[str] = None) -> list[dict]:
    """
    Fetch and parse jobs from an RSS feed.

    Args:
        feed_url: URL of the RSS feed.
        source_name: Human-readable source name.
        keywords: Expanded keyword list for filtering.

    Returns:
        List of normalized job dicts with relevance scores.
    """
    if not _HAS_FEEDPARSER:
        return []

    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"[RSS] Error parsing {source_name}: {e}")
        return []

    jobs = []

    for entry in feed.entries:
        try:
            title = entry.get("title", "N/A")
            summary = _clean_html(entry.get("summary", entry.get("description", "")))
            link = entry.get("link", "")

            # Extract company name — Remote OK uses 'company' field directly
            company = (
                entry.get("company", "")
                or entry.get("author", "")
                or entry.get("dc_creator", "")
                or ""
            ).strip()
            if not company or company == "N/A":
                company = ""

            # Extract tags safely
            tags_list = entry.get("tags", [])
            tags_str = ""
            if isinstance(tags_list, list):
                valid_tags = []
                for t in tags_list:
                    if isinstance(t, dict):
                        valid_tags.append(t.get("term", t.get("label", "")))
                    elif isinstance(t, str):
                        valid_tags.append(t)
                tags_str = " ".join(valid_tags)

            # Location
            location = entry.get("location", "Remote")

            # Calculate relevance if keywords provided
            if keywords:
                score = _relevance_score(title, summary, tags_str, keywords)
                if score == 0:
                    continue  # Skip completely irrelevant jobs
            else:
                score = 50  # Neutral score when no filter

            jobs.append({
                "title": title,
                "company": company,
                "location": location,
                "description": summary[:500],
                "link": link,
                "published": entry.get("published", str(datetime.now())),
                "source": source_name,
                "tags": tags_str,
                "relevance": score,
                "hash": _job_hash(title, link),
            })
        except Exception as e:
            # Skip malformed entry but continue
            continue

    # Sort by relevance (highest first)
    jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return jobs


# ═══════════════════════════════════════════════════════════════
#  API Fetching
# ═══════════════════════════════════════════════════════════════

def fetch_arbeitnow_jobs(keywords: list[str] = None) -> list[dict]:
    """Fetch jobs from Arbeitnow (free, no key required)."""
    if not _HAS_REQUESTS:
        return []

    try:
        resp = _requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[API] Arbeitnow error: {e}")
        return []

    jobs = []
    for item in data.get("data", []):
        title = item.get("title", "")
        desc = _clean_html(item.get("description", ""))
        company = item.get("company_name", "")
        tags_str = " ".join(item.get("tags", []))

        if keywords:
            score = _relevance_score(title, desc, tags_str, keywords)
            if score == 0:
                continue
        else:
            score = 50

        jobs.append({
            "title": title,
            "company": company,
            "location": item.get("location", "N/A"),
            "description": desc[:500],
            "link": item.get("url", ""),
            "published": item.get("created_at", ""),
            "source": "Arbeitnow",
            "tags": tags_str,
            "remote": item.get("remote", False),
            "relevance": score,
            "hash": _job_hash(title, item.get("url", "")),
        })

    jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return jobs


def fetch_themuse_jobs(keywords: list[str] = None, page: int = 1) -> list[dict]:
    """Fetch jobs from The Muse (free, no key required)."""
    if not _HAS_REQUESTS:
        return []

    try:
        resp = _requests.get(
            "https://www.themuse.com/api/public/jobs",
            params={"page": page, "descending": "true"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[API] The Muse error: {e}")
        return []

    jobs = []
    for item in data.get("results", []):
        title = item.get("name", "")
        company = item.get("company", {}).get("name", "")
        locations = ", ".join(
            loc.get("name", "") for loc in item.get("locations", [])
        ) or "N/A"
        desc = _clean_html(item.get("contents", ""))[:500]

        if keywords:
            score = _relevance_score(title, desc, "", keywords)
            if score == 0:
                continue
        else:
            score = 50

        link = f"https://www.themuse.com/jobs/{item.get('id', '')}"
        jobs.append({
            "title": title,
            "company": company,
            "location": locations,
            "description": desc,
            "link": link,
            "published": item.get("publication_date", ""),
            "source": "The Muse",
            "relevance": score,
            "hash": _job_hash(title, link),
        })

    jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return jobs


# ═══════════════════════════════════════════════════════════════
#  Remotive API (free, no key)
# ═══════════════════════════════════════════════════════════════

def fetch_remotive_jobs(keywords: list[str] = None, query: str = "") -> list[dict]:
    """Fetch jobs from Remotive."""
    if not _HAS_REQUESTS:
        return []
    try:
        params = {}
        if query:
            params["search"] = query
        resp = _requests.get("https://remotive.com/api/remote-jobs", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[API] Remotive error: {e}")
        return []

    jobs = []
    for item in data.get("jobs", []):
        title = item.get("title", "")
        desc = _clean_html(item.get("description", ""))[:500]
        company = item.get("company_name", "")
        tags_str = " ".join(item.get("tags", []))
        location = item.get("candidate_required_location", "Remote")

        if keywords:
            score = _relevance_score(title, desc, tags_str, keywords)
            if score == 0:
                continue
        else:
            score = 50

        link = item.get("url", "")
        jobs.append({
            "title": title,
            "company": company,
            "location": location or "Remote",
            "description": desc,
            "link": link,
            "published": item.get("publication_date", ""),
            "source": "Remotive",
            "tags": tags_str,
            "relevance": score,
            "hash": _job_hash(title, link),
        })

    jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return jobs


# ═══════════════════════════════════════════════════════════════
#  Jobicy API (free, no key)
# ═══════════════════════════════════════════════════════════════

def fetch_jobicy_jobs(keywords: list[str] = None, query: str = "") -> list[dict]:
    """Fetch jobs from Jobicy."""
    if not _HAS_REQUESTS:
        return []
    try:
        params = {"count": 50, "geo": "anywhere"}
        if query:
            params["tag"] = query.split()[0] if query.split() else ""
        resp = _requests.get("https://jobicy.com/api/v2/remote-jobs", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[API] Jobicy error: {e}")
        return []

    jobs = []
    for item in data.get("jobs", []):
        title = item.get("jobTitle", "")
        desc = _clean_html(item.get("jobDescription", ""))[:500]
        company = item.get("companyName", "")
        location = item.get("jobGeo", "Remote")
        job_type = item.get("jobType", "")

        if keywords:
            score = _relevance_score(title, desc, job_type, keywords)
            if score == 0:
                continue
        else:
            score = 50

        link = item.get("url", "")
        jobs.append({
            "title": title,
            "company": company,
            "location": location or "Remote",
            "description": desc,
            "link": link,
            "published": item.get("pubDate", ""),
            "source": "Jobicy",
            "tags": job_type,
            "relevance": score,
            "hash": _job_hash(title, link),
        })

    jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return jobs


# ═══════════════════════════════════════════════════════════════
#  Emploi.tn + TanitJobs (Tunisia) — HTML Scraping
# ═══════════════════════════════════════════════════════════════

def fetch_emploitn_jobs(keywords: list[str] = None, query: str = "") -> list[dict]:
    """Scrape jobs from Emploi.tn and TanitJobs (Tunisian job boards)."""
    if not _HAS_REQUESTS:
        return []

    jobs = []
    search_term = query.replace(" ", "+") if query else "informatique"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # ---------- Emploi.tn ----------
    try:
        url = f"https://www.emploi.tn/recherche?q={search_term}"
        resp = _requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        if _HAS_BS4:
            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.select(".card-job, .job-item, article.job, .result-item, .job-listing")
            if not cards:
                cards = soup.find_all("a", href=re.compile(r"/offre-emploi/|/job/"))

            for card in cards[:30]:
                try:
                    if card.name == "a":
                        title = card.get_text(strip=True)
                        link = card.get("href", "")
                        company, location = "", "Tunisia"
                    else:
                        title_el = card.select_one("h2, h3, .job-title, .title, a")
                        title = title_el.get_text(strip=True) if title_el else ""
                        link_el = card.select_one("a[href]")
                        link = link_el.get("href", "") if link_el else ""
                        company_el = card.select_one(".company, .company-name, .employer")
                        company = company_el.get_text(strip=True) if company_el else ""
                        loc_el = card.select_one(".location, .job-location, .city")
                        location = loc_el.get_text(strip=True) if loc_el else "Tunisia"

                    if not title or len(title) < 5:
                        continue
                    if link and not link.startswith("http"):
                        link = f"https://www.emploi.tn{link}"

                    score = _relevance_score(title, "", "", keywords) if keywords else 50
                    if keywords and score == 0:
                        score = 10  # Keep Tunisia jobs with low score

                    jobs.append({
                        "title": title, "company": company,
                        "location": location or "Tunisia", "description": "",
                        "link": link, "published": "", "source": "Emploi.tn",
                        "tags": "", "relevance": score,
                        "hash": _job_hash(title, link),
                    })
                except Exception:
                    continue
        else:
            links = re.findall(r'href="(/offre-emploi/[^"]+)"[^>]*>([^<]+)', resp.text)
            for href, title in links[:20]:
                title = title.strip()
                if len(title) < 5:
                    continue
                link = f"https://www.emploi.tn{href}"
                score = _relevance_score(title, "", "", keywords) if keywords else 50
                if keywords and score == 0:
                    score = 10
                jobs.append({
                    "title": title, "company": "", "location": "Tunisia",
                    "description": "", "link": link, "published": "",
                    "source": "Emploi.tn", "tags": "", "relevance": score,
                    "hash": _job_hash(title, link),
                })
    except Exception as e:
        print(f"[Scrape] Emploi.tn error: {e}")

    # ---------- TanitJobs ----------
    try:
        url2 = f"https://www.tanitjobs.com/search/?q={search_term}"
        resp2 = _requests.get(url2, headers=headers, timeout=15)
        if resp2.status_code == 200 and _HAS_BS4:
            soup2 = BeautifulSoup(resp2.text, "html.parser")
            cards2 = soup2.select(".job-item, .job-listing, article, .card")
            if not cards2:
                cards2 = soup2.find_all("a", href=re.compile(r"/job/|/offre/"))
            for card in cards2[:20]:
                try:
                    if card.name == "a":
                        title = card.get_text(strip=True)
                        link = card.get("href", "")
                        company = ""
                    else:
                        title_el = card.select_one("h2, h3, .title, a")
                        title = title_el.get_text(strip=True) if title_el else ""
                        link_el = card.select_one("a[href]")
                        link = link_el.get("href", "") if link_el else ""
                        company_el = card.select_one(".company, .employer")
                        company = company_el.get_text(strip=True) if company_el else ""

                    if not title or len(title) < 5:
                        continue
                    if link and not link.startswith("http"):
                        link = f"https://www.tanitjobs.com{link}"

                    score = _relevance_score(title, "", "", keywords) if keywords else 50
                    if keywords and score == 0:
                        score = 10

                    jobs.append({
                        "title": title, "company": company,
                        "location": "Tunisia", "description": "",
                        "link": link, "published": "",
                        "source": "TanitJobs", "tags": "", "relevance": score,
                        "hash": _job_hash(title, link),
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"[Scrape] TanitJobs error: {e}")

    jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return jobs


# ═══════════════════════════════════════════════════════════════
#  Unified Search Tool
# ═══════════════════════════════════════════════════════════════

def _deduplicate(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs by hash."""
    seen = set()
    unique = []
    for job in jobs:
        h = job.get("hash", "")
        if h and h not in seen:
            seen.add(h)
            unique.append(job)
        elif not h:
            unique.append(job)
    return unique


@tool
def job_search_tool(
    query: str,
    sources: str = "all",
    max_results: int = 25,
) -> dict:
    """
    Search for job postings across RSS feeds and APIs.

    Args:
        query: Job search keywords (e.g. "python developer", "IA", "data scientist").
        sources: Comma-separated sources or "all".
                 Options: Remote OK, We Work Remotely, Arbeitnow, The Muse,
                          Remotive, Jobicy, Emploi.tn.
        max_results: Maximum number of results to return (default 25).

    Returns:
        Dict with 'jobs' list and metadata.
    """
    keywords = _expand_query(query)

    requested = [s.strip().lower() for s in sources.split(",")] if sources != "all" else ["all"]
    all_jobs = []

    # --- RSS Feeds ---
    for feed_name, feed_url in RSS_FEEDS.items():
        if "all" not in requested and feed_name.lower() not in requested:
            continue
        cached = get_cached(query, feed_name)
        if cached is not None:
            all_jobs.extend(cached)
            continue
        jobs = fetch_rss_jobs(feed_url, feed_name, keywords)
        set_cache(query, feed_name, jobs)
        all_jobs.extend(jobs)

    # --- APIs ---
    def _fetch_source(name, fetcher, *args):
        if "all" in requested or name.lower() in requested:
            cached = get_cached(query, name)
            if cached is not None:
                all_jobs.extend(cached)
            else:
                jobs = fetcher(*args)
                set_cache(query, name, jobs)
                all_jobs.extend(jobs)

    _fetch_source("Arbeitnow", fetch_arbeitnow_jobs, keywords)
    _fetch_source("The Muse", fetch_themuse_jobs, keywords)
    _fetch_source("Remotive", fetch_remotive_jobs, keywords, query)
    _fetch_source("Jobicy", fetch_jobicy_jobs, keywords, query)

    # Tunisia sources — also triggered by "all"
    if "all" in requested or "emploi.tn" in requested or "tunisia" in requested or "tanitjobs" in requested:
        cached = get_cached(query, "Emploi.tn")
        if cached is not None:
            all_jobs.extend(cached)
        else:
            jobs = fetch_emploitn_jobs(keywords, query)
            set_cache(query, "Emploi.tn", jobs)
            all_jobs.extend(jobs)

    # --- Deduplicate, sort by relevance, limit ---
    all_jobs = _deduplicate(all_jobs)
    all_jobs.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    all_jobs = all_jobs[:max_results]

    return {
        "query": query,
        "keywords_used": keywords,
        "total_found": len(all_jobs),
        "sources_queried": sources,
        "jobs": [
            {
                "title": j.get("title", ""),
                "company": j.get("company", ""),
                "location": j.get("location", ""),
                "description": j.get("description", ""),
                "link": j.get("link", ""),
                "source": j.get("source", ""),
                "published": j.get("published", ""),
                "relevance": j.get("relevance", 0),
            }
            for j in all_jobs
        ],
    }
