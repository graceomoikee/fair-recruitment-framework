"""
Runs baseline models on a secondary dataset for validation.

Used to assess whether fairness patterns generalise beyond the primary dataset.
"""
import os

from src.data.loader_adult import load_adult
from src.models.baseline_registry import get_baseline_models
from src.evaluation.crossval import run_cv
from src.evaluation.aggregate import aggregate_cv
from src.core.logging import make_run_dir, save_metrics

SEED = 42


def main():
    # ---- Load Adult dataset ----
    ds = load_adult("data/raw/adult/adult.csv")

    models = get_baseline_models(ds.X, SEED)

    # ---- BASELINE CV ----
    baseline_records = run_cv(
        models=models,
        X=ds.X,
        y=ds.y,
        A=ds.A,
        seed=SEED,
        n_splits=5,
    )

    # IMPORTANT: label mitigation explicitly
    for r in baseline_records:
        r["mitigation"] = "baseline"

    # ---- Save outputs ----
    run_dir = make_run_dir("runs/secondary_validation")

    save_metrics(run_dir, baseline_records)

    baseline_summary = aggregate_cv(baseline_records)
    baseline_summary.to_csv(
        os.path.join(run_dir, "cv_summary_baseline.csv"),
        index=False,
    )

    print("Secondary validation (Adult Income, baseline only) complete.")
    print(baseline_summary)


if __name__ == "__main__":
    main()

