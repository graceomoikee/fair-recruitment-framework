from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    A: pd.Series
    feature_names: List[str]
    label_name: str
    protected_name: str
