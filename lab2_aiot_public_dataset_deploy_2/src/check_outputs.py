from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from src.data_utils import load_model_bundle

REQUIRED_FILES = [
    PROJECT_ROOT / "data" / "telemetry_clean.csv",
    PROJECT_ROOT / "data" / "feature_dataset.csv",
    PROJECT_ROOT / "models" / "occupancy_baseline.joblib",
    PROJECT_ROOT / "outputs" / "metrics.json",
    PROJECT_ROOT / "outputs" / "decision_log.csv",
    PROJECT_ROOT / "outputs" / "figures" / "01_co2_time_series.png",
    PROJECT_ROOT / "outputs" / "figures" / "02_confusion_matrix.png",
    PROJECT_ROOT / "outputs" / "figures" / "03_occupancy_probability.png",
]

if __name__ == "__main__":
    ok = True
    for path in REQUIRED_FILES:
        exists = path.exists() and path.stat().st_size > 0
        print(("OK  " if exists else "MISS") + str(path.relative_to(PROJECT_ROOT)))
        ok = ok and exists

    if not ok:
        raise SystemExit("Some required files are missing. Run the notebook or python src/run_training_pipeline.py first.")

    metrics = json.loads((PROJECT_ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    decision_log = pd.read_csv(PROJECT_ROOT / "outputs" / "decision_log.csv")
    bundle = load_model_bundle()

    required_decision_cols = [
        "timestamp", "occupancy_probability", "predicted_occupancy",
        "anomaly_score", "is_anomaly", "is_low_confidence", "confidence",
        "decision", "command_hint", "safety_note"
    ]
    missing_cols = [c for c in required_decision_cols if c not in decision_log.columns]
    print("\nMetrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Decision log rows:", len(decision_log))
    print("Model version:", bundle.get("model_version"))

    if missing_cols:
        raise SystemExit(f"Decision log missing columns: {missing_cols}")
    if len(decision_log) < 50:
        raise SystemExit("Decision log should have at least 50 rows.")
    if metrics.get("f1", 0) < 0.75:
        raise SystemExit("F1 is lower than expected for the baseline demo. Check preprocessing/model.")

    print("\nPROJECT CHECK PASSED: Notebook outputs and model artifacts are complete.")
