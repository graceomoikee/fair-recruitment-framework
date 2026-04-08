import pandas as pd


def aggregate_cv(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Columns that are NOT metrics
    non_metric_cols = {"fold", "model", "mitigation"}

    metric_cols = [c for c in df.columns if c not in non_metric_cols]

    agg = (
        df.groupby(["model", "mitigation"])[metric_cols]
        .agg(["mean", "std"])
    )

    agg.columns = ["_".join(col) for col in agg.columns]
    return agg.reset_index()


