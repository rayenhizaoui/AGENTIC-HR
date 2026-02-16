"""
Job Cache System

Supports Redis (if available) with automatic fallback to file-based JSON caching.
Configurable TTL to avoid redundant RSS/API calls.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "cache" / "jobs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TTL_MINUTES = int(os.getenv("REDIS_TTL_MINUTES", "30"))

# ── Redis support (optional) ──────────────────────────────────
_redis_client = None

try:
    import redis
    _redis_url = os.getenv("REDIS_URL", "")
    if _redis_url:
        _redis_client = redis.from_url(_redis_url, decode_responses=True)
        _redis_client.ping()
        print("[Cache] Redis connected ✅", file=sys.stderr)
except Exception as e:
    _redis_client = None
    print(f"[Cache] Redis unavailable, using file cache: {e}", file=sys.stderr)


def _cache_key(query: str, source: str) -> str:
    """Generate a unique cache key from query + source."""
    return hashlib.md5(f"{query.lower().strip()}:{source.lower().strip()}".encode()).hexdigest()


def get_cached(query: str, source: str, max_age_min: int = DEFAULT_TTL_MINUTES) -> list | None:
    """
    Return cached job results if they exist and are fresh.
    Tries Redis first, then falls back to file cache.

    Args:
        query: Search query.
        source: Source name (e.g. "Remote OK").
        max_age_min: Maximum cache age in minutes.

    Returns:
        List of job dicts, or None if cache is stale/missing.
    """
    key = _cache_key(query, source)

    # 1. Try Redis
    if _redis_client:
        try:
            raw = _redis_client.get(f"jobs:{key}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass

    # 2. Fallback to file cache
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["timestamp"])
        if datetime.now() - cached_at < timedelta(minutes=max_age_min):
            return data["jobs"]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    return None


def set_cache(query: str, source: str, jobs: list) -> None:
    """
    Save job results to cache (Redis + file for durability).

    Args:
        query: Search query.
        source: Source name.
        jobs: List of job dicts to cache.
    """
    key = _cache_key(query, source)

    # 1. Save to Redis (with TTL)
    if _redis_client:
        try:
            _redis_client.setex(
                f"jobs:{key}",
                DEFAULT_TTL_MINUTES * 60,  # TTL in seconds
                json.dumps(jobs, ensure_ascii=False),
            )
        except Exception:
            pass

    # 2. Always save to file (durability fallback)
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "source": source,
            "count": len(jobs),
            "jobs": jobs,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
