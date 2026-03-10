"""
ml/train.py

LOCAL RETRAINING TRIGGER — does NOT train the model locally.
Your PC cannot fit LGB+CAT+XGB with 900 estimators each.

What this file does when drift is detected:
  Option A (default): Trigger a Kaggle notebook run via Kaggle API
  Option B (fallback): Print instructions for manual Kaggle retrain

The Kaggle notebook (kaggle/train_and_upload.py) handles:
  - Pulling fresh data or using competition data
  - Training LGB + CAT + XGB voting classifier
  - Logging to WandB
  - Uploading new pipeline to HuggingFace
  - After upload, GitHub Actions Workflow A redeploys automatically

Called by: .github/workflows/workflow_b_retrain.yml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings

settings = get_settings()


def trigger_kaggle_retrain() -> bool:
    """
    Trigger your Kaggle notebook via the Kaggle API.

    Setup steps (one time):
      1. pip install kaggle
      2. Download kaggle.json from kaggle.com/your-account → API
      3. Place at C:/Users/YOU/.kaggle/kaggle.json  (Windows)
         or ~/.kaggle/kaggle.json                   (Linux/Mac)
      4. Set KAGGLE_USERNAME and KAGGLE_KEY in your .env

    Your Kaggle notebook must:
      - Be named: heart-disease-retrain  (or update kernel_slug below)
      - Have "Internet on" enabled in notebook settings
      - Have HF_TOKEN and WANDB_API_KEY added to Kaggle Secrets
    """
    try:
        import kaggle
        kaggle.api.authenticate()

        # Push a new run of your training notebook
        kaggle.api.kernels_push_cli(
            folder="kaggle",              # local kaggle/ folder with kernel-metadata.json
        )
        print("✅  Kaggle retrain triggered successfully")
        print("    Monitor at: https://www.kaggle.com/code/YOUR_USERNAME/heart-disease-retrain")
        print("    Once done, model will auto-upload to HuggingFace")
        print("    Workflow A will then redeploy automatically via repository_dispatch")
        return True

    except ImportError:
        print("⚠️  kaggle package not installed. Run: pip install kaggle")
        _print_manual_instructions()
        return False
    except Exception as exc:
        print(f"⚠️  Kaggle API trigger failed: {exc}")
        _print_manual_instructions()
        return False


def _print_manual_instructions():
    print("\n" + "="*60)
    print("MANUAL RETRAIN REQUIRED")
    print("="*60)
    print("1. Go to your Kaggle notebook:")
    print("   https://www.kaggle.com/code/YOUR_USERNAME/heart-disease-retrain")
    print("2. Click 'Run All'")
    print("3. Wait for completion (~20-30 mins)")
    print("4. New model will auto-upload to HuggingFace")
    print("5. Trigger Workflow A manually in GitHub Actions to redeploy")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("🔄  Drift detected — triggering Kaggle retrain ...")
    success = trigger_kaggle_retrain()
    sys.exit(0 if success else 1)