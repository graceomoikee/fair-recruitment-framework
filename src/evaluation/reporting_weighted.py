from src.metrics.performance import performance_metrics
from src.metrics.fairness import fairness_metrics


def evaluate_model_with_weights(
    pipeline,
    X_train, y_train, A_train,
    X_test, y_test, A_test,
    sample_weight,
) -> dict:
    """
    Evaluate a pipeline using instance-level sample weights
    (used for pre-processing mitigation such as reweighing).
    """

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight
    )

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    perf = performance_metrics(y_test, y_pred, y_score)
    fair = fairness_metrics(y_test, y_pred, A_test)

    return {**perf, **fair}

