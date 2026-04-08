from src.metrics.performance import performance_metrics
from src.metrics.fairness import fairness_metrics


def evaluate_expgrad(
    mitigator,
    X_test,
    y_test,
    A_test,
):
    """
    Evaluate an Exponentiated Gradient mitigator.
    """

    y_pred = mitigator.predict(X_test)

    # Some Fairlearn predictors expose predict_proba, some do not
    if hasattr(mitigator, "predict_proba"):
        y_score = mitigator.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use predictions as scores
        y_score = y_pred

    perf = performance_metrics(y_test, y_pred, y_score)
    fair = fairness_metrics(y_test, y_pred, A_test)

    return {**perf, **fair}

