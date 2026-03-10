"""
app/cache.py

All Upstash Redis logic.

1. Sliding-window rate limiter  — 3 tiers: free / pro / clinical
2. Prediction cache             — skip the 2700-estimator ensemble on repeat inputs

Rate limit key:  ratelimit:{tier}:{user_id}:{window_minute}
Cache key:       cache:heart:{sha256[:16]}
"""
import hashlib
import json
import time
from typing import Optional

from upstash_redis import Redis

from app.config import get_settings

settings = get_settings()

_redis = Redis(
    url=settings.upstash_redis_url,
    token=settings.upstash_redis_token,
)

TIER_LIMITS: dict[str, int] = {
    "free":     settings.rate_limit_free,
    "pro":      settings.rate_limit_pro,
    "clinical": settings.rate_limit_clinical,
}


# ── Rate Limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(user_id: str, tier: str = "free") -> tuple[bool, int]:
    """
    Returns (is_allowed, requests_remaining).
    Tier in key ensures Free→Pro upgrade mid-window starts fresh counter.
    """
    limit  = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    window = int(time.time() // 60)
    key    = f"ratelimit:{tier}:{user_id}:{window}"

    current = _redis.incr(key)
    _redis.expire(key, 90)
    remaining = max(0, limit - current)
    return current <= limit, remaining

# ── Prediction Cache ──────────────────────────────────────────────────────────

def _cache_key(features: dict) -> str:
    blob   = json.dumps(features, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return f"cache:heart:{digest}"


def get_cached_prediction(features: dict) -> Optional[dict]:
    raw = _redis.get(_cache_key(features))
    return json.loads(raw) if raw else None


def set_cached_prediction(features: dict, result: dict) -> None:
    _redis.setex(_cache_key(features), settings.cache_ttl_seconds, json.dumps(result))


def get_cache_stats() -> dict:
    try:
        info = _redis.info()
        return {
            "status":            "connected",
            "used_memory":       info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "ttl_seconds":       settings.cache_ttl_seconds,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}