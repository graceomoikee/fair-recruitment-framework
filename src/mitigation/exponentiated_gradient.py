from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    EqualizedOdds,
)
from sklearn.linear_model import LogisticRegression


def train_expgrad(
    X,
    y,
    A,
    constraint: str,
    seed: int,
):
    """
    Train an Exponentiated Gradient mitigator
    with either Demographic Parity or Equalized Odds.
    """

    base_estimator = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=seed,
    )

    if constraint == "dp":
        constraint_obj = DemographicParity()
    elif constraint == "eo":
        constraint_obj = EqualizedOdds()
    else:
        raise ValueError("constraint must be 'dp' or 'eo'")

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint_obj,
    )

    mitigator.fit(
        X,
        y,
        sensitive_features=A,
    )

    return mitigator

