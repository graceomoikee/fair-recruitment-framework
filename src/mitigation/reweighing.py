import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing


def to_aif360_dataset(y, A, label_name, protected_name):
    """
    Create an AIF360 BinaryLabelDataset using ONLY
    the label and a NUMERIC protected attribute.

    Gender is binarised here for compatibility with AIF360.
    """

    df = pd.concat(
        [y.rename(label_name), A.rename(protected_name)],
        axis=1,
    ).dropna()

    # Binarise protected attribute for AIF360
    df[protected_name] = df[protected_name].map(
        {"Male": 1, "Female": 0}
    )

    if df[protected_name].isna().any():
        raise ValueError("Unexpected protected attribute values after mapping.")

    return BinaryLabelDataset(
        df=df,
        label_names=[label_name],
        protected_attribute_names=[protected_name],
        favorable_label=1,
        unfavorable_label=0,
    )


def apply_reweighing(dataset, privileged_groups, unprivileged_groups):
    rw = Reweighing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
    )
    return rw.fit_transform(dataset)


def extract_weights(reweighed_dataset):
    return reweighed_dataset.instance_weights

