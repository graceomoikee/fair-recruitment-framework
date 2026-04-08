"""
Runs 5-fold stratified cross-validation using Exponentiated Gradient mitigation.

Evaluates fairness-performance trade-offs under:
- Demographic Parity (DP)
- Equalised Odds (EO)

Outputs aggregated metrics.
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_hr_analytics
from src.data.preprocessing import build_preprocessor
from src.mitigation.exponentiated_gradient import train_expgrad
from src.evaluation.reporting_expgrad import evaluate_expgrad
from src.evaluation.aggregate import aggregate_cv
from src.core.logging import make_run_dir, save_metrics

SEED = 42


def main():
    # ---------------------------
    # Load data
    # ---------------------------
    ds = load_hr_analytics("data/raw/HR_Analytics.csv")

    # CV setup
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    records = []

    # ---------------------------
    # Cross-validation loop
    # ---------------------------
    for fold, (train_idx, test_idx) in enumerate(skf.split(ds.X, ds.y), start=1):
        X_train, X_test = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
        y_train, y_test = ds.y.iloc[train_idx], ds.y.iloc[test_idx]
        A_train, A_test = ds.A.iloc[train_idx], ds.A.iloc[test_idx]

        # ---- PREPROCESS FEATURES (numeric only for EG) ----
        preprocessor = build_preprocessor(X_train)

        X_train_enc = preprocessor.fit_transform(X_train).toarray()
        X_test_enc = preprocessor.transform(X_test).toarray()

        # ---- Train EG with different constraints ----
        for constraint in ["dp", "eo"]:
            mitigator = train_expgrad(
                X_train_enc,
                y_train,
                A_train,
                constraint=constraint,
                seed=SEED,
            )

            metrics = evaluate_expgrad(
                mitigator,
                X_test_enc,
                y_test,
                A_test,
            )

            records.append(
                {
                    "fold": fold,
                    "model": "logreg",
                    "mitigation": f"expgrad_{constraint}",
                    **metrics,
                }
            )

    # ---------------------------
    # Save results
    # ---------------------------
    run_dir = make_run_dir()
    save_metrics(run_dir, records)

    summary = aggregate_cv(records)
    summary.to_csv(
        os.path.join(run_dir, "cv_summary_expgrad.csv"),
        index=False,
    )

    print("Exponentiated Gradient CV complete.")
    print("Results saved to:", run_dir)
    print(summary)


if __name__ == "__main__":
    main()

