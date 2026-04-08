from sklearn.model_selection import StratifiedKFold
from src.evaluation.reporting import evaluate_model


def run_cv(models: dict, X, y, A, seed: int, n_splits: int = 5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        A_train, A_test = A.iloc[train_idx], A.iloc[test_idx]

        for model_name, pipeline in models.items():
            metrics = evaluate_model(
                pipeline,
                X_train,
                y_train,
                A_train,
                X_test,
                y_test,
                A_test,
            )

            records.append(
                {
                    "fold": fold,
                    "model": model_name,
                    **metrics,
                }
            )

    return records

