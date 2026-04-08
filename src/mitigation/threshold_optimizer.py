from fairlearn.postprocessing import ThresholdOptimizer


def apply_threshold_optimizer(
    estimator,
    X_train,
    y_train,
    A_train,
):
    """
    Apply post-processing via Threshold Optimizer
    to enforce Equalized Odds on a pre-trained estimator.
    """

    post = ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",
        prefit=True,
    )

    post.fit(
        X_train,
        y_train,
        sensitive_features=A_train,
    )

    return post

