"""
ml/pipeline.py

LOCAL SIDE ONLY — no training happens here.
This file:
  1. Defines HeartDiseaseFeatureEngineer (must match what Kaggle trained with)
  2. Downloads the saved pipeline from HuggingFace on startup
  3. Exposes predict() for FastAPI

The pipeline saved on HuggingFace already contains:
  HeartDiseaseFeatureEngineer -> StandardScaler -> VotingClassifier(LGB+CAT+XGB)
So we just call pipeline.predict_proba() directly.

Training code lives in: kaggle/train_and_upload.py
"""
import joblib
from typing import Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from app.config import get_settings

settings = get_settings()

MODEL_FILENAME = "heart_disease_voting_pipeline.joblib"


# ── Feature Engineer (MUST match Kaggle notebook exactly) ────────────────────

class HeartDiseaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Copied verbatim from your Kaggle notebook.
    If you change features in Kaggle, update this class too.
    Both must stay in sync or predictions will be wrong.
    """

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        df = X.copy()

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        df["Age_BP"]          = df["Age"] * df["BP"]
        df["Age_Cholesterol"] = df["Age"] * df["Cholesterol"]
        df["BP_Cholesterol"]  = df["BP"]  * df["Cholesterol"]
        df["MaxHR_Age"]       = df["Max HR"] / (df["Age"] + 1)
        df["ST_Age"]          = df["ST depression"] * df["Age"]

        # Binary risk flags
        df["High_BP"]          = (df["BP"]           > 140).astype(int)
        df["High_Cholesterol"] = (df["Cholesterol"]  > 240).astype(int)
        df["Low_MaxHR"]        = (df["Max HR"]        < 100).astype(int)
        df["High_ST"]          = (df["ST depression"] > 2  ).astype(int)

        # Composite risk score (0-4)
        df["Risk_Score"] = (
            df["High_BP"]
            + df["High_Cholesterol"]
            + df["Low_MaxHR"]
            + df["High_ST"]
        )

        # Age group buckets
        df["Age_Group"] = (
            pd.cut(
                df["Age"],
                bins=[0, 40, 50, 60, 100],
                labels=[0, 1, 2, 3],
            ).astype(int)
        )

        return df


# ── Model Manager ─────────────────────────────────────────────────────────────

class ModelManager:
    """
    Downloads pipeline from HuggingFace once on startup.
    Keeps it in RAM — never reloads per request.
    """

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.model_version: str = "unknown"
        self._loaded: bool = False

    def load(self) -> None:
        """Called once in FastAPI lifespan on startup."""
        print("Downloading model from HuggingFace ...")

        hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=MODEL_FILENAME,
            token=settings.hf_token,
            local_dir="/tmp",
        )

        # Register the transformer class so joblib can find it during unpickling.
        # Kaggle saved the pipeline with this class under __main__ so we must
        # make it visible there before joblib.load() runs.
        import __main__
        __main__.HeartDiseaseFeatureEngineer = HeartDiseaseFeatureEngineer

        self.pipeline = joblib.load(f"/tmp/{MODEL_FILENAME}")

        # Get HuggingFace commit SHA for version tracking
        try:
            api  = HfApi()
            info = api.repo_info(repo_id=settings.hf_repo_id, token=settings.hf_token)
            self.model_version = (getattr(info, "sha", None) or "unknown")[:8]
        except Exception:
            self.model_version = "unknown"

        self._loaded = True
        print(f"Model loaded — version {self.model_version}")

    def is_ready(self) -> bool:
        return self._loaded and self.pipeline is not None

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 0.25:
            return "low"
        if score < 0.50:
            return "moderate"
        if score < 0.75:
            return "high"
        return "critical"

    def predict(self, features: dict) -> dict:
        """
        features = raw dict with original Kaggle column names.
        e.g. {"Age": 54, "Max HR": 92, "ST depression": 2.8 ...}
        Pipeline handles feature engineering + scaling internally.
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded yet")

        df         = pd.DataFrame([features])
        prediction = int(self.pipeline.predict(df)[0])
        proba      = self.pipeline.predict_proba(df)[0]
        risk_score = float(proba[1])
        confidence = float(max(proba))

        return {
            "prediction":    prediction,
            "risk_score":    round(risk_score, 4),
            "risk_level":    self._risk_level(risk_score),
            "confidence":    round(confidence, 4),
            "model_version": self.model_version,
        }

    def warm_predict(self) -> bool:
        """Called by /health/ready — runs a real dummy inference to confirm model works."""
        try:
            dummy = {
                "Age": 54, "Sex": 1,
                "Chest pain type": 0,
                "BP": 145, "Cholesterol": 255,
                "FBS over 120": 0,
                "EKG results": 1,
                "Max HR": 92,
                "Exercise angina": 1,
                "ST depression": 2.8,
                "Slope of ST": 1,
                "Number of vessels fluro": 2,
                "Thallium": 2,
            }
            return "risk_score" in self.predict(dummy)
        except Exception:
            return False


# Singleton — one instance shared across all FastAPI async workers
model_manager = ModelManager()