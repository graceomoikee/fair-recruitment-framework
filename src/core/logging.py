import json
import os
from datetime import datetime
import pandas as pd


def make_run_dir(base="runs"):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_metrics(run_dir: str, records: list[dict]):
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

