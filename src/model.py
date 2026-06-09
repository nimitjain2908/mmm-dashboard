import numpy as np
import pandas as pd
import joblib
import os
from src.types import MMMModel
from typing import Dict, List
from sklearn.linear_model import Ridge
from src.transformations import transform_all_channels, add_seasonality_features, CHANNELS
from src.optimiser import run_optimisation, fit_final_model, calculate_vif, CONTROL_COLS

CHANNELS = ['tv', 'digital', 'search', 'email', 'ooh']
MODEL_PATH = 'outputs/mmm_model.pkl'


def extract_coefficients(model: Ridge, feature_cols: List[str]) -> Dict[str, float]:
    """
    Extract named coefficients from fitted Ridge model.
    Maps each feature name to its regression coefficient.
    """
    return dict(zip(feature_cols, model.coef_))


def calculate_channel_roi(
    df: pd.DataFrame,
    df_transformed: pd.DataFrame,
    coefficients: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate ROI per channel.

    ROI = Total revenue contribution / Total actual spend

    Revenue contribution = coefficient × sum of transformed spend
    scaled to revenue units (×500,000 from data generation scaling).
    """
    roi = {}
    scale_factor = 500000

    for channel in CHANNELS:
        coef         = coefficients.get(f'{channel}_transformed', 0)
        transformed  = df_transformed[f'{channel}_transformed'].values
        contribution = coef * transformed.sum()
        total_spend  = df[f'{channel}_spend'].sum()

        if total_spend > 0:
            roi[channel] = round(contribution / total_spend, 3)
        else:
            roi[channel] = 0.0

    return roi


def build_mmm_model(df: pd.DataFrame, n_iterations: int = 2) -> MMMModel:
    """
    Full pipeline: optimise parameters → fit final model → package results.
    """
    # Step 1: Find best transformation parameters
    print("Step 1: Optimising transformation parameters...")
    best_params = run_optimisation(df, n_iterations=n_iterations)

    # Step 2: Fit final model with best parameters
    print("\nStep 2: Fitting final model...")
    model, r2, mape, y_pred, df_transformed, feature_cols = fit_final_model(
        df, best_params
    )

    # Step 3: Extract coefficients
    coefficients = extract_coefficients(model, feature_cols)
    print("\nRegression coefficients:")
    for feat, coef in coefficients.items():
        print(f"  {feat:<30} {coef:>10.2f}")

    # Step 4: Calculate ROI
    channel_roi = calculate_channel_roi(df, df_transformed, coefficients)
    print("\nChannel ROI (revenue per £1 spent):")
    for channel, roi in sorted(channel_roi.items(), key=lambda x: -x[1]):
        print(f"  {channel:<12} {roi:>8.2f}x")

    # Step 5: Package into MMMModel
    mmm = MMMModel(
        model=model,
        best_params=best_params,
        feature_cols=feature_cols,
        coefficients=coefficients,
        intercept=model.intercept_,
        r2=r2,
        mape=mape,
        channel_roi=channel_roi,
        df_transformed=df_transformed,
        y_pred=y_pred
    )

    return mmm


def save_model(mmm: MMMModel, path: str = MODEL_PATH):
    """Save the fitted MMMModel object to disk."""
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(mmm, path)
    print(f"\nModel saved to {path}")


def load_model(path: str = MODEL_PATH) -> MMMModel:
    """Load a previously saved MMMModel from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv('data/mmm_data.csv')

    # Build and save model
    mmm = build_mmm_model(df, n_iterations=2)
    save_model(mmm)

    print(f"\nModel summary:")
    print(f"  R²:        {mmm.r2:.4f}")
    print(f"  MAPE:      {mmm.mape:.2f}%")
    print(f"  Intercept: £{mmm.intercept:,.0f}")
    print(f"\nBest parameters found:")
    for ch, params in mmm.best_params.items():
        print(f"  {ch:<12} decay={params['decay']:.1f}  k={params['k']:.1f}  alpha={params['alpha']:.1f}")