"""
Runs 5-fold stratified cross-validation with Reweighing (pre-processing mitigation).

Adjusts sample weights to reduce bias before model training.

Evaluates impact on fairness and predictive performance.
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_hr_analytics
from src.models.baseline_registry import get_baseline_models
from src.mitigation.reweighing import (
    to_aif360_dataset,
    apply_reweighing,
    extract_weights,
)
from src.evaluation.reporting_weighted import evaluate_model_with_weights
from src.evaluation.aggregate import aggregate_cv
from src.core.logging import make_run_dir, save_metrics

SEED = 42


def main():
    # ---------------------------
    # Load data
    # ---------------------------
    ds = load_hr_analytics("data/raw/HR_Analytics.csv")

    # Baseline models (unchanged)
    models = get_baseline_models(ds.X, SEED)

    # AIF360 group definitions
    privileged_groups = [{"Gender": 1}]
    unprivileged_groups = [{"Gender": 0}]

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

        # ----- Reweighing on TRAINING data only (labels + protected attribute only) -----
        train_df = pd.concat(
            [y_train.rename("Attrition"), A_train.rename("Gender")],
            axis=1,
        ).dropna()

        y_train_clean = train_df["Attrition"]
        A_train_clean = train_df["Gender"]

        # Create AIF360 dataset (NO FEATURES)
        train_aif = to_aif360_dataset(
            y_train_clean,
            A_train_clean,
            label_name="Attrition",
            protected_name="Gender",
        )

        # Apply reweighing
        reweighed = apply_reweighing(
            train_aif,
            privileged_groups,
            unprivileged_groups,
        )

        sample_weights = extract_weights(reweighed)

        # ----- Train & evaluate models with sample weights -----
        for model_name, pipeline in models.items():
            metrics = evaluate_model_with_weights(
                pipeline,
                X_train.loc[y_train_clean.index],  # align features to cleaned labels
                y_train_clean,
                A_train_clean,
                X_test,
                y_test,
                A_test,
                sample_weight=sample_weights,
            )

            records.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "mitigation": "reweighing",
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
        os.path.join(run_dir, "cv_summary_reweighing.csv"),
        index=False,
    )

    print("Reweighing CV complete.")
    print("Results saved to:", run_dir)
    print(summary)


if __name__ == "__main__":
    main()

