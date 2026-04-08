import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_hr_analytics
from src.data.preprocessing import build_preprocessor
from src.mitigation.reweighing import (
    to_aif360_dataset,
    apply_reweighing,
    extract_weights,
)
from src.mitigation.exponentiated_gradient import train_expgrad
from src.evaluation.reporting_expgrad import evaluate_expgrad
from src.evaluation.aggregate import aggregate_cv
from src.core.logging import make_run_dir, save_metrics
from src.core.seed import set_global_seed

SEED = 42
N_SPLITS = 5


def main():
    set_global_seed(SEED)

    # Load data
    ds = load_hr_analytics("data/raw/HR_Analytics.csv")

    privileged_groups = [{"Gender": "Male"}]
    unprivileged_groups = [{"Gender": "Female"}]

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )

    records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(ds.X, ds.y), start=1):
        X_train, X_test = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
        y_train, y_test = ds.y.iloc[train_idx], ds.y.iloc[test_idx]
        A_train, A_test = ds.A.iloc[train_idx], ds.A.iloc[test_idx]

        # ---------- REWEIGHING (TRAIN ONLY) ----------
        train_df = pd.concat(
            [y_train.rename("Attrition"), A_train.rename("Gender")],
            axis=1
        ).dropna()

        y_train_clean = train_df["Attrition"]
        A_train_clean = train_df["Gender"]

        train_aif = to_aif360_dataset(
            y_train_clean,
            A_train_clean,
            label_name="Attrition",
            protected_name="Gender",
        )

        reweighed = apply_reweighing(
            train_aif,
            privileged_groups,
            unprivileged_groups,
        )

        sample_weight = extract_weights(reweighed)

        # ---------- FEATURE PREPROCESSING ----------
        preprocessor = build_preprocessor(X_train)

        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc = preprocessor.transform(X_test)

        # Convert sparse → dense (required by Fairlearn)
        if hasattr(X_train_enc, "toarray"):
            X_train_enc = X_train_enc.toarray()
            X_test_enc = X_test_enc.toarray()

        # ---------- EXPONENTIATED GRADIENT (DP) ----------
        mitigator = train_expgrad(
            X_train_enc,
            y_train,
            A_train,
            constraint="dp",
            seed=SEED,
            sample_weight=sample_weight,
        )

        metrics = evaluate_expgrad(
            mitigator,
            X_test_enc,
            y_test,
            A_test,
        )

        records.append({
            "fold": fold,
            "model": "logreg",
            "mitigation": "reweighing_expgrad_dp",
            **metrics,
        })

    # ---------- SAVE RESULTS ----------
    run_dir = make_run_dir()
    save_metrics(run_dir, records)

    summary = aggregate_cv(records)
    summary.to_csv(
        os.path.join(run_dir, "cv_summary_multistage.csv"),
        index=False
    )

    print("Multi-stage CV complete.")
    print("Results saved to:", run_dir)
    print(summary)


if __name__ == "__main__":
    main()

