"""
app/config.py

Single source of truth for all settings.
Reads from .env file once, cached forever via lru_cache.
Every other file does: from app.config import get_settings
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Your existing services ────────────────────────────────────────────────
    database_url:          str
    upstash_redis_url:     str
    upstash_redis_token:   str
    hf_token:              str
    hf_repo_id:            str              # e.g. your-username/heart-disease-model
    wandb_api_key:         str
    wandb_project:         str  = "heart-disease-mlops"
    wandb_entity:          str  = ""

    # ── App ───────────────────────────────────────────────────────────────────
    app_env:               str  = "production"
    secret_key:            str  = "change-me"

    # ── Cache (12h — clinical data can change daily) ──────────────────────────
    cache_ttl_seconds:     int  = 43200

    # ── Drift detection ───────────────────────────────────────────────────────
    min_drift_batch_size:  int   = 50
    psi_threshold:         float = 0.2
    ks_pvalue_threshold:   float = 0.05
    ks_min_features:       int   = 2

    # ── Rate limits ───────────────────────────────────────────────────────────
    rate_limit_free:       int  = 5
    rate_limit_pro:        int  = 20
    rate_limit_clinical:   int  = 100

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()