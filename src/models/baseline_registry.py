from src.models.baseline import (
    get_logistic_regression_pipeline,
    get_random_forest_pipeline,
    get_gradient_boosting_pipeline,
)


def get_baseline_models(X, seed: int):
    return {
        "logreg": get_logistic_regression_pipeline(X),
        "rf": get_random_forest_pipeline(X, seed),
        "gb": get_gradient_boosting_pipeline(X, seed),
    }

