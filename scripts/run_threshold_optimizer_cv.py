"""
Runs post-processing mitigation using Threshold Optimisation.

Applies group-specific decision thresholds to enforce Equalised Odds.

Evaluates fairness improvements without retraining the model.
"""
import os
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_hr_analytics
from src.data.preprocessing import build_preprocessor
from src.models.baseline import get_logistic_regression_pipeline
from src.mitigation.threshold_optimizer import apply_threshold_optimizer
from src.evaluation.reporting_postprocess import evaluate_postprocessed
from src.evaluation.aggregate import aggregate_cv
from src.core.logging import make_run_dir, save_metrics

SEED = 42


def main():
    ds = load_hr_analytics("data/raw/HR_Analytics.csv")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(ds.X, ds.y), start=1):
        X_train, X_test = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
        y_train, y_test = ds.y.iloc[train_idx], ds.y.iloc[test_idx]
        A_train, A_test = ds.A.iloc[train_idx], ds.A.iloc[test_idx]

        # ----- Train baseline Logistic Regression -----
        pipeline = get_logistic_regression_pipeline(ds.X)
        pipeline.fit(X_train, y_train)

        # Extract fitted estimator and preprocessor
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]

        # Transform features
        X_train_enc = preprocessor.transform(X_train).toarray()
        X_test_enc = preprocessor.transform(X_test).toarray()

        # ----- Apply Threshold Optimizer -----
        post = apply_threshold_optimizer(
            estimator,
            X_train_enc,
            y_train,
            A_train,
        )

        metrics = evaluate_postprocessed(
            post,
            X_test_enc,
            y_test,
            A_test,
        )

        records.append({
            "fold": fold,
            "model": "logreg",
            "mitigation": "threshold_eo",
            **metrics,
        })

    run_dir = make_run_dir()
    save_metrics(run_dir, records)

    summary = aggregate_cv(records)
    summary.to_csv(
        os.path.join(run_dir, "cv_summary_threshold.csv"),
        index=False,
    )

    print("Threshold Optimizer CV complete.")
    print("Results saved to:", run_dir)
    print(summary)


if __name__ == "__main__":
    main()

