"""
tests/test_core.py

Run: pytest tests/ -v --cov=app --cov=ml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Sample data ───────────────────────────────────────────────────────────────

def _sample() -> dict:
    return {
        "Age": 54.0, "Sex": 1,
        "Chest Pain Type": 0,
        "BP": 145.0, "Cholesterol": 255.0,
        "Fasting BS": 0, "Resting ECG": 1,
        "Max HR": 92.0, "Exercise Angina": 1,
        "ST depression": 2.8, "Slope": 1,
        "Num Vessels": 2, "Thal": 2,
    }


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame([_sample()])


# ══════════════════════════════════════════════════════════════════════════════
# HeartDiseaseFeatureEngineer
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:

    @pytest.fixture
    def fe(self):
        from ml.pipeline import HeartDiseaseFeatureEngineer
        e = HeartDiseaseFeatureEngineer()
        e.fit(_sample_df())
        return e

    def test_fit_returns_self(self, fe):
        assert fe.fit(_sample_df()) is fe

    def test_drops_id_column(self, fe):
        df = _sample_df()
        df["id"] = 999
        assert "id" not in fe.transform(df).columns

    def test_no_error_without_id(self, fe):
        assert "id" not in fe.transform(_sample_df()).columns

    # Interaction terms
    def test_age_bp(self, fe):
        r = fe.transform(_sample_df())
        assert r["Age_BP"].iloc[0] == pytest.approx(54.0 * 145.0)

    def test_age_cholesterol(self, fe):
        r = fe.transform(_sample_df())
        assert r["Age_Cholesterol"].iloc[0] == pytest.approx(54.0 * 255.0)

    def test_bp_cholesterol(self, fe):
        r = fe.transform(_sample_df())
        assert r["BP_Cholesterol"].iloc[0] == pytest.approx(145.0 * 255.0)

    def test_maxhr_age(self, fe):
        r = fe.transform(_sample_df())
        assert r["MaxHR_Age"].iloc[0] == pytest.approx(92.0 / 55.0)

    def test_st_age(self, fe):
        r = fe.transform(_sample_df())
        assert r["ST_Age"].iloc[0] == pytest.approx(2.8 * 54.0)

    # Binary flags
    def test_high_bp_true(self, fe):
        assert fe.transform(_sample_df())["High_BP"].iloc[0] == 1

    def test_high_bp_false(self, fe):
        df = _sample_df(); df["BP"] = 130
        assert fe.transform(df)["High_BP"].iloc[0] == 0

    def test_high_cholesterol_true(self, fe):
        assert fe.transform(_sample_df())["High_Cholesterol"].iloc[0] == 1

    def test_high_cholesterol_false(self, fe):
        df = _sample_df(); df["Cholesterol"] = 200
        assert fe.transform(df)["High_Cholesterol"].iloc[0] == 0

    def test_low_maxhr_true(self, fe):
        assert fe.transform(_sample_df())["Low_MaxHR"].iloc[0] == 1

    def test_low_maxhr_false(self, fe):
        df = _sample_df(); df["Max HR"] = 150
        assert fe.transform(df)["Low_MaxHR"].iloc[0] == 0

    def test_high_st_true(self, fe):
        assert fe.transform(_sample_df())["High_ST"].iloc[0] == 1

    def test_high_st_false(self, fe):
        df = _sample_df(); df["ST depression"] = 1.0
        assert fe.transform(df)["High_ST"].iloc[0] == 0

    # Risk score
    def test_risk_score_max(self, fe):
        assert fe.transform(_sample_df())["Risk_Score"].iloc[0] == 4

    def test_risk_score_zero(self, fe):
        df = pd.DataFrame([{
            "Age": 35, "Sex": 0, "Chest Pain Type": 3,
            "BP": 110, "Cholesterol": 180, "Fasting BS": 0,
            "Resting ECG": 0, "Max HR": 175, "Exercise Angina": 0,
            "ST depression": 0.5, "Slope": 2, "Num Vessels": 0, "Thal": 2,
        }])
        assert fe.transform(df)["Risk_Score"].iloc[0] == 0

    # Age groups
    def test_age_group_under_40(self, fe):
        df = _sample_df(); df["Age"] = 35
        assert fe.transform(df)["Age_Group"].iloc[0] == 0

    def test_age_group_40_50(self, fe):
        df = _sample_df(); df["Age"] = 45
        assert fe.transform(df)["Age_Group"].iloc[0] == 1

    def test_age_group_50_60(self, fe):
        df = _sample_df(); df["Age"] = 55
        assert fe.transform(df)["Age_Group"].iloc[0] == 2

    def test_age_group_over_60(self, fe):
        df = _sample_df(); df["Age"] = 65
        assert fe.transform(df)["Age_Group"].iloc[0] == 3

    def test_dict_input_works(self, fe):
      result = fe.transform(pd.DataFrame([_sample()]))
      assert isinstance(result, pd.DataFrame)
      assert "Age_BP" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# Risk Level
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskLevel:

    @pytest.fixture
    def mgr(self):
        from ml.pipeline import ModelManager
        return ModelManager()

    def test_low(self, mgr):        assert mgr._risk_level(0.10) == "low"
    def test_moderate(self, mgr):   assert mgr._risk_level(0.35) == "moderate"
    def test_high(self, mgr):       assert mgr._risk_level(0.60) == "high"
    def test_critical(self, mgr):   assert mgr._risk_level(0.85) == "critical"

    def test_boundary_low_moderate(self, mgr):
        assert mgr._risk_level(0.249) == "low"
        assert mgr._risk_level(0.250) == "moderate"

    def test_boundary_moderate_high(self, mgr):
        assert mgr._risk_level(0.499) == "moderate"
        assert mgr._risk_level(0.500) == "high"

    def test_boundary_high_critical(self, mgr):
        assert mgr._risk_level(0.749) == "high"
        assert mgr._risk_level(0.750) == "critical"


# ══════════════════════════════════════════════════════════════════════════════
# Cache Keys
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheKeys:

    def _key(self, f):
        import hashlib, json
        return "cache:heart:" + hashlib.sha256(
            json.dumps(f, sort_keys=True).encode()
        ).hexdigest()[:16]

    def test_order_independent(self):
        assert self._key({"Age": 54, "BP": 145}) == self._key({"BP": 145, "Age": 54})

    def test_value_sensitive(self):
        assert self._key({"Age": 54}) != self._key({"Age": 55})

    def test_correct_prefix(self):
        assert self._key({"Age": 40}).startswith("cache:heart:")


# ══════════════════════════════════════════════════════════════════════════════
# Rate Limit Keys
# ══════════════════════════════════════════════════════════════════════════════

class TestRateLimitKeys:

    def test_tier_isolation(self):
        keys = [f"ratelimit:{t}:user1:1000" for t in ("free", "pro", "clinical")]
        assert len(set(keys)) == 3

    def test_window_rotation(self):
        assert "ratelimit:free:u:1000" != "ratelimit:free:u:1001"

    def test_user_isolation(self):
        assert "ratelimit:pro:alice:1000" != "ratelimit:pro:bob:1000"


# ══════════════════════════════════════════════════════════════════════════════
# Drift Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestDrift:

    def test_psi_identical_near_zero(self):
        from monitor.drift_monitor import compute_psi
        d = np.random.normal(130, 15, 500)
        assert compute_psi(d, d) < 0.01

    def test_psi_shifted_high(self):
        from monitor.drift_monitor import compute_psi
        assert compute_psi(
            np.random.normal(120, 10, 500),
            np.random.normal(200, 10, 500)
        ) > 0.2

    def test_ks_no_drift(self):
        from monitor.drift_monitor import run_ks_gate
        ref  = pd.DataFrame({"Age": np.random.normal(54, 9, 300), "BP": np.random.normal(130, 18, 300)})
        prod = pd.DataFrame({"Age": np.random.normal(54, 9, 100), "BP": np.random.normal(130, 18, 100)})
        fired, _, _ = run_ks_gate(ref, prod)
        assert not fired

    def test_ks_obvious_drift(self):
        from monitor.drift_monitor import run_ks_gate
        ref  = pd.DataFrame({"Age": np.random.normal(30, 3, 300), "BP": np.random.normal(110, 5, 300)})
        prod = pd.DataFrame({"Age": np.random.normal(75, 3, 100), "BP": np.random.normal(185, 5, 100)})
        fired, _, _ = run_ks_gate(ref, prod)
        assert fired

    def test_psi_gate_stable(self):
        from monitor.drift_monitor import run_psi_gate
        data = np.random.normal(246, 40, 300)
        ref  = pd.DataFrame({"Cholesterol": data})
        prod = pd.DataFrame({"Cholesterol": np.random.normal(246, 40, 100)})
        fired, _ = run_psi_gate(ref, prod)
        assert not fired

    def test_psi_gate_drifted(self):
        from monitor.drift_monitor import run_psi_gate
        ref  = pd.DataFrame({"Cholesterol": np.random.normal(200, 20, 300)})
        prod = pd.DataFrame({"Cholesterol": np.random.normal(400, 20, 100)})
        fired, _ = run_psi_gate(ref, prod)
        assert fired