## Heart Disease ML — README

A small production-style ML service that predicts heart disease risk using a voting ensemble (LightGBM + CatBoost + XGBoost) trained in a Kaggle notebook and served via a FastAPI application.

This README explains project structure, how the system works end-to-end, how to run and test it locally, and where to look for common operational tasks (retraining, drift detection, deployment).

---

## Table of contents

- Project overview
- Architecture and components
- Configuration (.env)
- Running locally
- Docker & Kubernetes
- Retraining & drift monitoring
- Tests
- Development notes & gotchas
- Troubleshooting

## Project overview

This repository contains an inference API that:

- Serves a pre-trained pipeline downloaded from HuggingFace on startup.
- Performs feature engineering (same code as training) and inference.
- Caches predictions in Upstash Redis to speed up repeated requests.
- Rate-limits requests (three tiers: free / pro / clinical).
- Stores every request (raw input + prediction + metadata) in a Postgres (Neon) table for auditing and drift monitoring.
- Runs a drift monitor that compares recent production inputs to the training distribution; if drift is confirmed, it triggers a retrain workflow (Kaggle notebook) which uploads a new model to HuggingFace and (optionally) redeploys.

Key files:

- `app/main.py`        — FastAPI app, endpoints and request flow
- `ml/pipeline.py`     — Feature engineering + model download + predict wrapper
- `app/cache.py`       — Upstash Redis caching and rate limiting
- `app/database.py`    — Async DB models & helpers (heart_assessments, retrain_log)
- `monitor/drift_monitor.py` — Dual-gate drift detection (KS + PSI)
- `ml/train.py`        — Retrain trigger (calls Kaggle API or prints manual steps)
- `k8s/deployment.yaml` — Example Kubernetes deployment
- `Dockerfile`         — Container image for the API
- `tests/`             — Unit tests covering feature engineering, cache keys, drift functions

## Architecture & request flow

1. Client POSTs to `/assess` with the Kaggle-style input columns.
2. FastAPI checks the rate limit (Upstash Redis sliding window).
3. Looks up cache key based on sorted JSON of features. If present, returns cached result and logs the request using `app.database.save_assessment` (cache_hit=True).
4. If no cache, the singleton `ModelManager` (in `ml/pipeline.py`) runs the pipeline.predict() — the pipeline includes feature engineering saved with the model.
5. Prediction result is cached (TTL configured in `app.config`) and persisted to Postgres (heart_assessments table) for later drift analysis.

Health endpoints useful for orchestration:

- `GET /health/startup` — waits for model to load (startup probe)
- `GET /health/live`    — process liveness
- `GET /health/ready`   — warm inference (readiness probe)
- `GET /health/cache`   — cache + rate-limit diagnostics

## Configuration (.env)

All runtime settings come from environment variables via `app.config.get_settings()` (Pydantic/BaseSettings). The important variables you must provide in a `.env` file or environment:

- `database_url`         — Postgres/Neon connection string
- `upstash_redis_url`    — Upstash Redis URL
- `upstash_redis_token`  — Upstash Redis token
- `hf_token`             — HuggingFace token with repo read access
- `hf_repo_id`           — HuggingFace repo id where the pipeline artifact is stored (e.g. `user/heart-disease-model`)
- `wandb_api_key`        — (optional) Weights & Biases API key for training runs

There are other optional variables (rate limits, cache TTL, drift thresholds). See `app/config.py` for defaults and explanations.

Security: keep tokens secret. Do not commit `.env` into VCS.

## How to run locally

1. Create & activate a Python environment (recommended Python 3.11+).
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Provide a `.env` file at repository root with at least the required keys (see Configuration section).

4. Start the API (development):

```powershell
uvicorn app.main:app --reload --port 8000
```

Notes:

- On startup the app will try to download `heart_disease_voting_pipeline.joblib` from the HuggingFace repo configured by `hf_repo_id`. If the token/ID are missing or the model is not uploaded, the app will fail to load the model.
- The feature-engineering code used by the pipeline is defined in `ml/pipeline.py` (class `HeartDiseaseFeatureEngineer`). It must stay in sync with the training notebook or predictions will be incorrect.

Example request JSON for `/assess` (keys must match the Pydantic schema in `app/main.py`):

```json
{
  "Age": 54.0,
  "Sex": 1,
  "Chest Pain Type": 0,
  "BP": 145.0,
  "Cholesterol": 255.0,
  "Fasting BS": 0,
  "Resting ECG": 1,
  "Max HR": 92.0,
  "Exercise Angina": 1,
  "ST depression": 2.8,
  "Slope": 1,
  "Num Vessels": 2,
  "Thal": 2,
  "user_id": "anonymous",
  "user_tier": "free"
}
```

## Docker & Kubernetes

- `Dockerfile` is included for building a container image. The image runs the FastAPI app and expects configuration via environment variables.
- `k8s/deployment.yaml` contains a sample Kubernetes Deployment and probes wired against `/health/*` endpoints.

When deployed in Kubernetes, use `startupProbe` → `readyProbe` → `livenessProbe` in that order to ensure the model finishes downloading before traffic is allowed.

## Drift detection & retraining

Drift monitor: `monitor/drift_monitor.py` — a dual-gate approach:

- Gate 1: multiple-feature Kolmogorov–Smirnov (KS) tests (counts how many numeric features have p-values below the configured threshold).
- Gate 2: population stability index (PSI) averaged across numeric features.
- Both gates must trigger before a retrain is considered confirmed.

When both gates fire, the monitor writes an entry to the `retrain_log` table and exits with code `2`. This exit code is used by GitHub Actions (Workflow B) to trigger the retraining job.

Retraining: `ml/train.py` contains the retrain trigger that either uses the Kaggle API to push a notebook run, or prints manual instructions if the Kaggle package isn't installed. The heavy model training (LGB/CAT/XGB ensemble, logging to WandB, and uploading the resulting pipeline artifact to HuggingFace) is performed in the Kaggle notebook (not included here).

Audit trail: retrain checks and outcomes are stored in the `retrain_log` table inside Postgres.

## Tests

Run the unit tests with pytest:

```powershell
pytest tests/ -v --cov=app --cov=ml
```

The test suite covers feature engineering, cache key behavior, rate-limit key formats, and drift statistics helper functions.

## Development notes & gotchas

- Keep `HeartDiseaseFeatureEngineer` in `ml/pipeline.py` identical to the code used during training. Mismatches will silently break prediction quality.
- The saved pipeline (joblib) must include the same class name and location, so the API registers the class on startup before joblib.loads (see `ml/pipeline.py` where `__main__.HeartDiseaseFeatureEngineer` is set).
- Model artifact name: `heart_disease_voting_pipeline.joblib` — this is downloaded on startup by `ModelManager.load()` into `/tmp`.
- Cache keys are deterministic: cache key is a sha256 of JSON-dumped features with sorted keys. That keeps key generation order-independent (see `app/cache.py`).

## Troubleshooting

- Model download fails on startup — verify `hf_token` and `hf_repo_id` in your `.env` and ensure the artifact exists in the repo. Check logs printed by `ModelManager.load()`.
- Model not loaded / 503 from `/assess` — the readiness or startup probe might still be running. The model could be missing or unpickling failed because the feature-engineer class isn't available; see the code that registers it before joblib.load.
- Redis errors — check `GET /health/cache` for connection problems and verify Upstash credentials.
- Database errors — verify `database_url` and that Neon/Postgres allows connections from the app host. The app uses SQLAlchemy async engine.

## Where to go next / operational suggestions

- Add automated CI that runs unit tests and black/flake checks.
- Add a small integration test that starts the app with a mocked model and cache backends.
- Consider adding schema validation or typed pydantic models for the DB layer to strengthen drift-sensitive comparisons.

---

If you'd like, I can:

- Add a sample `.env.example` with the minimal variables.
- Add a simple Docker-compose file for local development (Postgres + Redis + app) so you can run end-to-end locally.

Happy to generate either—tell me which you'd like next.
