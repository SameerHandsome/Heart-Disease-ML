"""
app/database.py

All Neon Postgres logic.

Tables:
  heart_assessments  — every prediction request stored with raw pre-transform
                       features as JSON. Drift monitor reads from this table.
  retrain_log        — permanent audit trail of every drift check and retrain.
"""
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, Integer, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import get_settings

settings = get_settings()

_engine = create_async_engine(
    settings.database_url
        .replace("postgresql://", "postgresql+asyncpg://")
        .replace("?sslmode=require", ""),
    pool_size=5,
    max_overflow=10,
    echo=False,
    connect_args={"ssl": True},
    pool_recycle=300,        # recycle connections every 5 min
    pool_pre_ping=True,      
)

AsyncSessionLocal = sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


class HeartAssessment(Base):
    """
    One row per /assess call.
    raw_input = original Kaggle columns as JSON (before feature engineering).
    Evidently drift monitor compares this against training distribution.
    """
    __tablename__ = "heart_assessments"

    id          = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = Column(String,  nullable=False)
    user_tier   = Column(String,  default="free")
    raw_input   = Column(Text,    nullable=False)   # JSON: Age, BP, Cholesterol etc
    prediction  = Column(Integer, nullable=False)   # 0=Absence, 1=Presence
    risk_score  = Column(Float,   nullable=False)   # P(Presence) 0.0-1.0
    risk_level  = Column(String,  nullable=False)   # low|moderate|high|critical
    confidence  = Column(Float,   nullable=False)
    model_ver   = Column(String,  nullable=False)   # HuggingFace commit SHA
    cache_hit   = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class RetrainLog(Base):
    """Permanent audit trail — never deleted."""
    __tablename__ = "retrain_log"

    id            = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    triggered_by  = Column(String)
    ks_score      = Column(Float)
    psi_score     = Column(Float)
    batch_size    = Column(Integer)
    retrain_fired = Column(Boolean)
    new_model_ver = Column(String)
    outcome       = Column(String)
    notes         = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def init_db():
    """Create tables on startup if they don't exist."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields session, auto-closes after request."""
    async with AsyncSessionLocal() as session:
        yield session


# ── Helpers ───────────────────────────────────────────────────────────────────

async def save_assessment(db: AsyncSession, record: HeartAssessment) -> None:
    db.add(record)
    await db.commit()


async def fetch_recent_raw_inputs(db: AsyncSession, limit: int = 200) -> list[dict]:
    """Used by drift monitor to get recent production data."""
    result = await db.execute(
        text("SELECT raw_input FROM heart_assessments ORDER BY created_at DESC LIMIT :limit"),
        {"limit": limit},
    )
    return [json.loads(row[0]) for row in result.fetchall()]