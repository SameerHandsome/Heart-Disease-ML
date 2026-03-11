# ============================================================
# KAGGLE NOTEBOOK — Heart Disease Voting Classifier
# Competition: playground-series-s6e2
#
# HOW TO USE:
#   1. Create a new Kaggle notebook named: heart-disease-retrain
#   2. Paste this entire file into it
#   3. Enable Internet in notebook settings
#   4. Add secrets in Kaggle (Add-ons → Secrets):
#        HF_TOKEN      = your HuggingFace token
#        WANDB_API_KEY = your WandB key
#   5. Run All
# ============================================================

import pandas as pd
import numpy as np
import warnings
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')


# ============================================================
# STEP 1 — Setup WandB and HuggingFace
# ============================================================

from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

import wandb
wandb.login(key=secrets.get_secret("WANDB_API_KEY"))

run = wandb.init(
    project="heart-disease-mlops",
    job_type="retrain",
    tags=["kaggle", "voting-classifier", "playground-s6e2"],
    config={
        "n_estimators":  900,
        "max_depth":     7,
        "learning_rate": 0.05,
        "subsample":     0.8,
        "voting":        "soft",
        "models":        ["lgb", "cat", "xgb"],
    }
)

from huggingface_hub import HfApi
hf_token = secrets.get_secret("HF_TOKEN")
HF_REPO   = "sameerhandsome12/heart-disease-model"   # ← CHANGE THIS TO YOUR HF USERNAME

api = HfApi()
api.create_repo(
    repo_id=HF_REPO,
    token=hf_token,
    exist_ok=True,
    private=True,
)
print(f"✅ HuggingFace repo ready: {HF_REPO}")


# ============================================================
# STEP 2 — Custom Transformer
# ============================================================

class HeartDiseaseFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        df['Age_BP']          = df['Age'] * df['BP']
        df['Age_Cholesterol'] = df['Age'] * df['Cholesterol']
        df['BP_Cholesterol']  = df['BP']  * df['Cholesterol']
        df['MaxHR_Age']       = df['Max HR'] / (df['Age'] + 1)
        df['ST_Age']          = df['ST depression'] * df['Age']

        df['High_BP']          = (df['BP']           > 140).astype(int)
        df['High_Cholesterol'] = (df['Cholesterol']  > 240).astype(int)
        df['Low_MaxHR']        = (df['Max HR']        < 100).astype(int)
        df['High_ST']          = (df['ST depression'] > 2  ).astype(int)

        df['Risk_Score'] = (
            df['High_BP'] + df['High_Cholesterol'] +
            df['Low_MaxHR'] + df['High_ST']
        )

        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 40, 50, 60, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return df


# ============================================================
# STEP 3 — Load Data
# ============================================================

print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s6e2/train.csv')

X_raw = train.drop(columns=['Heart Disease'])
y_raw = (train['Heart Disease'] == 'Presence').astype(int)

print(f"  Rows: {len(train)}")
print(f"  Positive rate: {y_raw.mean():.2%}")

wandb.log({
    "total_samples": len(train),
    "positive_rate": float(y_raw.mean()),
})


# ============================================================
# STEP 4 — Define Models
# ============================================================

lgb_model = lgb.LGBMClassifier(
    n_estimators=900, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
cat_model = cb.CatBoostClassifier(
    iterations=900, depth=7, learning_rate=0.05,
    random_seed=42, verbose=False,
)
xgb_model = xgb.XGBClassifier(
    n_estimators=900, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='auc',
    n_jobs=-1, use_label_encoder=False,
)

voting_clf = VotingClassifier(
    estimators=[("lgb", lgb_model), ("cat", cat_model), ("xgb", xgb_model)],
    voting="soft",
)

mlops_pipeline = Pipeline([
    ('feature_engineer', HeartDiseaseFeatureEngineer()),
    ('scaler',           StandardScaler()),
    ('classifier',       voting_clf),
])


# ============================================================
# STEP 5 — Cross Validation
# ============================================================

skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

print("\nStarting Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_raw), 1):
    print(f"========== Fold {fold} ==========")
    fold_pipeline = clone(mlops_pipeline)
    fold_pipeline.fit(X_raw.iloc[train_idx], y_raw.iloc[train_idx])
    val_preds = fold_pipeline.predict_proba(X_raw.iloc[val_idx])[:, 1]
    auc = roc_auc_score(y_raw.iloc[val_idx], val_preds)
    auc_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.5f}")
    wandb.log({f"cv_fold_{fold}_auc": auc})

mean_auc = np.mean(auc_scores)
print(f"\nFinal CV AUC: {mean_auc:.5f} ± {np.std(auc_scores):.5f}")
wandb.log({
    "cv_mean_auc": float(mean_auc),
    "cv_std_auc":  float(np.std(auc_scores)),
})


# ============================================================
# STEP 6 — Fit on Full Dataset and Save
# ============================================================

print("\nFitting on entire dataset...")
mlops_pipeline.fit(X_raw, y_raw)

MODEL_FILENAME = "heart_disease_voting_pipeline.joblib"
joblib.dump(mlops_pipeline, MODEL_FILENAME)
print(f"Saved: {MODEL_FILENAME}")


# ============================================================
# STEP 7 — Upload to HuggingFace
# ============================================================

print("\nUploading to HuggingFace...")
commit = api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=HF_REPO,
    token=hf_token,
    commit_message=f"Retrain | cv_auc={mean_auc:.5f}",
)

print(f"✅ Uploaded to https://huggingface.co/{HF_REPO}")
wandb.log({"outcome": "uploaded_to_hf"})
wandb.finish()

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"CV AUC:  {mean_auc:.5f}")
print(f"Model:   https://huggingface.co/{HF_REPO}")
print("="*50)