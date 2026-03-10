"""
app/main.py

FastAPI application.
Model is NOT trained here — it is downloaded from HuggingFace on startup.
Training happens on Kaggle (see kaggle/train_and_upload.py).

Endpoints:
  POST /assess         — heart disease risk prediction
  GET  /health/startup — K8s startup probe
  GET  /health/live    — K8s liveness probe
  GET  /health/ready   — K8s readiness probe (runs warm inference)
  GET  /health/cache   — Redis diagnostics

Request flow:
  ① Rate limit check  (Upstash ~5ms)
  ② Cache lookup      (Upstash ~10ms)
  ③ Model inference   (LGB+CAT+XGB pipeline ~80-300ms)
  → Always log raw input to Neon for drift detection
"""
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import check_rate_limit, get_cache_stats, get_cached_prediction, set_cached_prediction
from app.config import get_settings
from app.database import HeartAssessment, get_db, init_db, save_assessment
from ml.pipeline import model_manager

settings = get_settings()


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()          
    model_manager.load()      
    yield


app = FastAPI(
    title="Heart Disease Assessment API",
    description="Predicts heart disease risk. Model trained on Kaggle, served from HuggingFace.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class AssessmentRequest(BaseModel):
    """
    Your exact Kaggle columns.
    Ranges based on the Cleveland heart disease dataset (playground-series-s6e2).
    """
    Age:              float = Field(..., ge=18,  le=100)
    Sex:              int   = Field(..., ge=0,   le=1,   description="0=Female 1=Male")
    Chest_Pain_Type:  int   = Field(..., ge=0,   le=3,   alias="Chest Pain Type")
    BP:               float = Field(..., ge=80,  le=220, description="Resting BP mmHg")
    Cholesterol:      float = Field(..., ge=100, le=600, description="Serum cholesterol mg/dL")
    Fasting_BS:       int   = Field(..., ge=0,   le=1,   alias="Fasting BS")
    Resting_ECG:      int   = Field(..., ge=0,   le=2,   alias="Resting ECG")
    Max_HR:           float = Field(..., ge=60,  le=220, alias="Max HR")
    Exercise_Angina:  int   = Field(..., ge=0,   le=1,   alias="Exercise Angina")
    ST_depression:    float = Field(..., ge=0.0, le=7.0, alias="ST depression")
    Slope:            int   = Field(..., ge=0,   le=2)
    Num_Vessels:      int   = Field(..., ge=0,   le=4,   alias="Num Vessels")
    Thal:             int   = Field(..., ge=0,   le=3)

    # Not sent to model — used for rate limiting and logging only
    user_id:   str = Field(default="anonymous")
    user_tier: str = Field(default="free", pattern="^(free|pro|clinical)$")

    model_config = {"populate_by_name": True}


class AssessmentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  
    request_id:    str
    prediction:    int
    risk_score:    float
    risk_level:    str
    confidence:    float
    model_version: str
    cache_hit:     bool


def _to_model_features(req: AssessmentRequest) -> dict:
    """
    Map to the exact column names the model was trained with.
    These match the actual Kaggle dataset column names.
    """
    return {
        "Age":                    req.Age,
        "Sex":                    req.Sex,
        "Chest pain type":        req.Chest_Pain_Type,
        "BP":                     req.BP,
        "Cholesterol":            req.Cholesterol,
        "FBS over 120":           req.Fasting_BS,
        "EKG results":            req.Resting_ECG,
        "Max HR":                 req.Max_HR,
        "Exercise angina":        req.Exercise_Angina,
        "ST depression":          req.ST_depression,
        "Slope of ST":            req.Slope,
        "Number of vessels fluro": req.Num_Vessels,
        "Thallium":               req.Thal,
    }

# ── Prediction ────────────────────────────────────────────────────────────────

@app.post("/assess", response_model=AssessmentResponse)
async def assess(
    request: AssessmentRequest,
    db: AsyncSession = Depends(get_db),
) -> AssessmentResponse:

    request_id = str(uuid.uuid4())
    features   = _to_model_features(request)

    # ① Rate limit
    allowed, remaining = check_rate_limit(request.user_id, request.user_tier)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error":               "Rate limit exceeded",
                "tier":                request.user_tier,
                "retry_after_seconds": 60,
            },
        )

    # ② Cache
    cached = get_cached_prediction(features)
    if cached:
        await save_assessment(db, HeartAssessment(
            id=request_id, user_id=request.user_id, user_tier=request.user_tier,
            raw_input=json.dumps(features),
            prediction=cached["prediction"], risk_score=cached["risk_score"],
            risk_level=cached["risk_level"], confidence=cached["confidence"],
            model_ver=cached["model_version"], cache_hit=True,
        ))
        return AssessmentResponse(**cached, cache_hit=True, request_id=request_id)

    # ③ Model inference
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model still loading. Retry shortly.")

    result = model_manager.predict(features)
    set_cached_prediction(features, result)

    await save_assessment(db, HeartAssessment(
        id=request_id, user_id=request.user_id, user_tier=request.user_tier,
        raw_input=json.dumps(features),
        prediction=result["prediction"], risk_score=result["risk_score"],
        risk_level=result["risk_level"], confidence=result["confidence"],
        model_ver=result["model_version"], cache_hit=False,
    ))

    return AssessmentResponse(**result, cache_hit=False, request_id=request_id)


# ── Health Probes ─────────────────────────────────────────────────────────────

@app.get("/health/startup")
async def startup_probe():
    """Gives joblib 100s to download + deserialise before liveness takes over."""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Still loading")
    return {"status": "started"}


@app.get("/health/live")
async def liveness_probe():
    """Process alive check — keep fast, no dependencies."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_probe():
    """Runs real warm inference — K8s won't send traffic until this passes."""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not model_manager.warm_predict():
        raise HTTPException(status_code=503, detail="Warm inference failed")
    return {"status": "ready", "model_version": model_manager.model_version}


@app.get("/health/cache")
async def cache_health():
    return {
        "cache":       get_cache_stats(),
        "rate_limits": {
            "free_rpm":     settings.rate_limit_free,
            "pro_rpm":      settings.rate_limit_pro,
            "clinical_rpm": settings.rate_limit_clinical,
        },
    }