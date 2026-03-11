"""
ml/train.py

Triggered by drift detection in Workflow B.
Pushes a new run of your Kaggle notebook via Kaggle API.
The notebook trains LGB+CAT+XGB and uploads to HuggingFace.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def trigger_kaggle_retrain() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()

        username = os.environ.get("KAGGLE_USERNAME", "")
        kernel_slug = f"{username}/heart-disease-retrain"

        print(f"📤  Pushing kernel: {kernel_slug}")

        # Pull existing kernel metadata first
        api.kernels_push("kaggle")

        print(f"✅  Kaggle retrain triggered")
        print(f"    Monitor: https://www.kaggle.com/code/{kernel_slug}")
        return True

    except Exception as exc:
        print(f"⚠️  Kaggle API trigger failed: {exc}")
        _print_manual_instructions()
        return False


def _print_manual_instructions():
    username = os.environ.get("KAGGLE_USERNAME", "YOUR_USERNAME")
    print("\n" + "=" * 60)
    print("MANUAL RETRAIN REQUIRED")
    print("=" * 60)
    print(f"1. Go to: https://www.kaggle.com/code/{username}/heart-disease-retrain")
    print("2. Click 'Run All'")
    print("3. Wait ~20-30 mins")
    print("4. Model uploads to HuggingFace automatically")
    print("5. Trigger Workflow A manually in GitHub Actions")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("🔄  Drift detected — triggering Kaggle retrain ...")
    success = trigger_kaggle_retrain()
    sys.exit(0 if success else 1)
