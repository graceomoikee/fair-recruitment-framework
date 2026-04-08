from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.preprocessing import build_preprocessor


def get_logistic_regression_pipeline(X):
    preprocessor = build_preprocessor(X)

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    return pipeline
def get_random_forest_pipeline(X, seed: int):
    preprocessor = build_preprocessor(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


def get_gradient_boosting_pipeline(X, seed: int):
    preprocessor = build_preprocessor(X)

    clf = GradientBoostingClassifier(
        random_state=seed
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

