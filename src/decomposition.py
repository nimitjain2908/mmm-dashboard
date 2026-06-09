import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.transformations import CHANNELS
from src.types import MMMModel

CONTROL_COLS = ['week_index', 'season_sin', 'season_cos']


def decompose_revenue(mmm) -> pd.DataFrame:
    """
    Decompose total revenue into contributions from each channel
    plus baseline (intercept + seasonality + trend).

    For each week:
        channel_contribution = coefficient × transformed_spend
        baseline             = intercept + seasonality + trend components

    Returns DataFrame with columns:
        week, actual_revenue, predicted_revenue,
        tv_contrib, digital_contrib, search_contrib,
        email_contrib, ooh_contrib, baseline_contrib
    """
    df_t   = mmm.df_transformed.copy()
    coeffs = mmm.coefficients
    model  = mmm.model

    result = pd.DataFrame()
    result['week']           = pd.to_datetime(df_t['week'])
    result['actual_revenue'] = df_t['revenue'].values
    result['predicted_revenue'] = mmm.y_pred

    # Channel contributions
    # Channel contributions
    for channel in CHANNELS:
        col   = f'{channel}_transformed'
        coef  = coeffs.get(col, 0)
        result[f'{channel}_contrib'] = coef * df_t[col].values

    # Baseline = intercept + seasonality + trend
    baseline = np.full(len(df_t), model.intercept_)
    for col in CONTROL_COLS:
        coef      = coeffs.get(col, 0)
        baseline += coef * df_t[col].values

    # Floor baseline at zero — negative portions go to residual
    baseline_floored           = np.maximum(baseline, 0)
    baseline_negative          = baseline - baseline_floored
    result['baseline_contrib'] = baseline_floored

    # Residual — unexplained portion plus any negative baseline
    total_media = result[[f'{ch}_contrib' for ch in CHANNELS]].sum(axis=1)
    total_explained = total_media + baseline_floored
    result['residual'] = result['actual_revenue'] - total_explained + baseline_negative

    return result


def get_total_contributions(decomp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise total revenue contribution and percentage share per channel.
    Used for the ROI summary table in the dashboard.
    """
    contrib_cols = [f'{ch}_contrib' for ch in CHANNELS] + ['baseline_contrib']
    totals       = decomp_df[contrib_cols].sum()
    total_rev    = decomp_df['actual_revenue'].sum()

    summary = pd.DataFrame({
        'Channel':      [ch.replace('_contrib', '').title() for ch in contrib_cols],
        'Contribution': totals.values.astype(int),
        'Share_%':      (totals.values / total_rev * 100).round(1)
    })

    return summary.sort_values('Contribution', ascending=False).reset_index(drop=True)


def get_weekly_decomposition(decomp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return weekly decomposition in format ready for Plotly stacked bar chart.
    Melts wide format (one column per channel) to long format
    (one row per channel per week) which Plotly px.bar expects.
    """
    value_cols = [f'{ch}_contrib' for ch in CHANNELS] + ['baseline_contrib']
    
    melted = decomp_df.melt(
        id_vars=['week'],
        value_vars=value_cols,
        var_name='Channel',
        value_name='Revenue'
    )
    melted['Channel'] = melted['Channel'].str.replace('_contrib', '').str.title()
    return melted


def validate_decomposition(decomp_df: pd.DataFrame) -> dict:
    """
    Sanity checks on decomposition quality.
    The sum of all contributions should approximately equal actual revenue.
    """
    contrib_cols  = [f'{ch}_contrib' for ch in CHANNELS] + ['baseline_contrib', 'residual']
    total_contrib = decomp_df[contrib_cols].sum(axis=1)
    total_actual  = decomp_df['actual_revenue']

    max_error_pct = ((total_contrib - total_actual).abs() / total_actual * 100).max()
    avg_error_pct = ((total_contrib - total_actual).abs() / total_actual * 100).mean()

    return {
        'max_reconstruction_error_%': round(max_error_pct, 4),
        'avg_reconstruction_error_%': round(avg_error_pct, 4),
        'decomposition_valid':        max_error_pct < 1.0
    }


if __name__ == "__main__":
    from src.model import load_model

    mmm      = load_model()
    decomp   = decompose_revenue(mmm)
    summary  = get_total_contributions(decomp)
    val      = validate_decomposition(decomp)

    print("Revenue Decomposition Summary:")
    print(summary.to_string(index=False))

    print(f"\nDecomposition validation:")
    print(f"  Max reconstruction error: {val['max_reconstruction_error_%']}%")
    print(f"  Avg reconstruction error: {val['avg_reconstruction_error_%']}%")
    print(f"  Valid: {val['decomposition_valid']}")

    print(f"\nFirst 5 weeks of decomposition:")
    cols = ['week', 'actual_revenue', 'predicted_revenue'] + \
           [f'{ch}_contrib' for ch in CHANNELS] + ['baseline_contrib']
    print(decomp[cols].head().to_string(index=False))