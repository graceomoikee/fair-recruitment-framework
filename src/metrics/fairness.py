from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equal_opportunity_difference,
)


def fairness_metrics(y_true, y_pred, sensitive_features) -> dict:
    return {
        "dp_diff": float(
            demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            )
        ),
        "di_ratio": float(
            demographic_parity_ratio(
                y_true, y_pred, sensitive_features=sensitive_features
            )
        ),
        "eo_diff": float(
            equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            )
        ),
        "eop_diff": float(
            equal_opportunity_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            )
        ),
    }

