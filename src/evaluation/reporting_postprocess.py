from src.metrics.performance import performance_metrics
from src.metrics.fairness import fairness_metrics


def evaluate_postprocessed(
    postprocessor,
    X_test,
    y_test,
    A_test,
):
    """
    Evaluate a post-processed model (Threshold Optimizer).

    Note:
    ThresholdOptimizer does NOT expose predict_proba().
    Probabilities are obtained via the internal _pmf_predict method.
    """

    # Hard predictions
    y_pred = postprocessor.predict(
        X_test,
        sensitive_features=A_test,
    )

    # Probabilities via internal Fairlearn API
    y_score = postprocessor._pmf_predict(
        X_test,
        sensitive_features=A_test,
    )[:, 1]

    perf = performance_metrics(y_test, y_pred, y_score)
    fair = fairness_metrics(y_test, y_pred, A_test)

    return {**perf, **fair}

