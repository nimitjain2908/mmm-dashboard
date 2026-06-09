from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class MMMModel:
    """
    Container for the fitted MMM model and all associated metadata.
    Stored in a dedicated module to ensure consistent pickle/unpickle paths.
    """
    model:          Ridge
    best_params:    Dict[str, Dict[str, float]]
    feature_cols:   List[str]
    coefficients:   Dict[str, float]
    intercept:      float
    r2:             float
    mape:           float
    channel_roi:    Dict[str, float]
    df_transformed: pd.DataFrame
    y_pred:         np.ndarray