import pandas as pd
from .schemas import DatasetBundle


def load_hr_analytics(csv_path: str) -> DatasetBundle:
    df = pd.read_csv(csv_path)

    label_name = "Attrition"
    protected_name = "Gender"

    if label_name not in df.columns:
        raise ValueError(f"Missing label column: {label_name}")
    if protected_name not in df.columns:
        raise ValueError(f"Missing protected column: {protected_name}")

    y = df[label_name].copy()
    A = df[protected_name].copy()

    # Drop label + protected from features
    X = df.drop(columns=[label_name, protected_name]).copy()

    # Basic validity checks (do not “clean” beyond this)
    if y.isna().any():
        raise ValueError("y contains missing values.")
    if A.isna().any():
        raise ValueError("A contains missing values.")

    # Normalise label values to 0/1 if needed (common in this dataset: Yes/No)
    if y.dtype == object:
        y = y.map({"Yes": 1, "No": 0})

    if set(y.unique()) - {0, 1}:
        raise ValueError(
            f"y must be binary 0/1 after mapping. Got: {sorted(y.unique())}"
        )

    return DatasetBundle(
        X=X,
        y=y.astype(int),
        A=A,
        feature_names=list(X.columns),
        label_name=label_name,
        protected_name=protected_name,
    )

