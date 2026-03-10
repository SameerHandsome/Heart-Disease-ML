"""
monitor/drift_monitor.py

Dual-gate drift detection.
Reads from heart_assessments table in Neon Postgres.
Compares recent production inputs against training reference distribution.

Both gates must fire independently before retrain triggers.
Single gate = too many false alarms from sampling noise.

Exit codes (read by Workflow B):
  0 = no drift
  1 = error
  2 = both gates fired → trigger retrain
"""
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings

settings = get_settings()


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI < 0.1 stable | 0.1-0.2 monitor | > 0.2 drift."""
    breaks      = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    exp_cnt, _  = np.histogram(expected, bins=breaks)
    act_cnt, _  = np.histogram(actual,   bins=breaks)
    exp_pct     = (exp_cnt + 1e-6) / len(expected)
    act_pct     = (act_cnt + 1e-6) / len(actual)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ── Gate 1: KS Test ───────────────────────────────────────────────────────────

def run_ks_gate(ref: pd.DataFrame, prod: pd.DataFrame) -> tuple[bool, dict, float]:
    num_cols = ref.select_dtypes(include=[np.number]).columns
    shared   = [c for c in num_cols if c in prod.columns]

    results, drifted = {}, 0
    for col in shared:
        _, pval      = ks_2samp(ref[col].dropna(), prod[col].dropna())
        results[col] = round(float(pval), 4)
        if pval < settings.ks_pvalue_threshold:
            drifted += 1

    fired    = drifted >= settings.ks_min_features
    fraction = drifted / max(len(shared), 1)
    print(f"  Gate 1 (KS):  {drifted}/{len(shared)} features drifted → {'FIRED 🔴' if fired else 'passed ✅'}")
    return fired, results, fraction


# ── Gate 2: PSI ───────────────────────────────────────────────────────────────

def run_psi_gate(ref: pd.DataFrame, prod: pd.DataFrame) -> tuple[bool, float]:
    num_cols = ref.select_dtypes(include=[np.number]).columns
    shared   = [c for c in num_cols if c in prod.columns]
    scores   = [compute_psi(ref[c].dropna().values, prod[c].dropna().values) for c in shared]
    avg_psi  = float(np.mean(scores)) if scores else 0.0
    fired    = avg_psi > settings.psi_threshold
    print(f"  Gate 2 (PSI): avg={avg_psi:.4f} → {'FIRED 🔴' if fired else 'passed ✅'}")
    return fired, avg_psi


# ── Data ──────────────────────────────────────────────────────────────────────

def load_reference(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT raw_input FROM heart_assessments ORDER BY created_at ASC LIMIT 500"
        )).fetchall()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def load_production(engine, limit: int) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT raw_input FROM heart_assessments ORDER BY created_at DESC LIMIT :n"
        ), {"n": limit}).fetchall()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def write_log(engine, ks_score, psi_score, batch_size, retrain_fired, outcome, notes):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO retrain_log
              (id, triggered_by, ks_score, psi_score, batch_size,
               retrain_fired, outcome, notes, created_at)
            VALUES (:id,'schedule',:ks,:psi,:batch,:fired,:outcome,:notes,NOW())
        """), {
            "id": str(uuid.uuid4()), "ks": ks_score, "psi": psi_score,
            "batch": batch_size, "fired": retrain_fired,
            "outcome": outcome, "notes": notes,
        })
        conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"\n🫀  Heart Disease Drift Monitor")
    print(f"    {datetime.utcnow():%Y-%m-%d %H:%M UTC}  |  batch={batch_size}\n")

    engine = create_engine(settings.database_url)

    if batch_size < settings.min_drift_batch_size:
        msg = f"Batch {batch_size} < min {settings.min_drift_batch_size}. Skipping."
        print(f"⚠️   {msg}")
        write_log(engine, 0, 0, batch_size, False, "skipped", msg)
        sys.exit(0)

    ref_df  = load_reference(engine)
    prod_df = load_production(engine, batch_size)

    if len(prod_df) < settings.min_drift_batch_size:
        msg = f"Only {len(prod_df)} production rows. Skipping."
        print(f"⚠️   {msg}")
        write_log(engine, 0, 0, len(prod_df), False, "skipped", msg)
        sys.exit(0)

    ks_fired,  ks_detail, ks_score = run_ks_gate(ref_df, prod_df)
    psi_fired, avg_psi             = run_psi_gate(ref_df, prod_df)
    both   = ks_fired and psi_fired
    outcome = "retrain_triggered" if both else "no_drift"
    notes   = f"KS_fired={ks_fired} PSI_fired={psi_fired} detail={json.dumps(ks_detail)}"

    print(f"\n{'🚨 DRIFT CONFIRMED — triggering Kaggle retrain' if both else '✅ No drift — no action needed'}\n")
    write_log(engine, ks_score, avg_psi, len(prod_df), both, outcome, notes)
    sys.exit(2 if both else 0)


if __name__ == "__main__":
    main()