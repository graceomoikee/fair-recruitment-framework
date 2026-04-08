import pandas as pd
from types import SimpleNamespace

def load_adult(csv_path):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=columns,
        skipinitialspace=True
    )

    # Label
    y = df["income"].map({">50K": 1, "<=50K": 0})

    # Protected attribute (Gender)
    A = df["sex"]

    # Features (drop label only)
    X = df.drop(columns=["income"])

    return SimpleNamespace(X=X, y=y, A=A)

