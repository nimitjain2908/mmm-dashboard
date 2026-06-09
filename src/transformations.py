import numpy as np
import pandas as pd
from typing import Dict, Tuple

CHANNELS = ['tv', 'digital', 'search', 'email', 'ooh']


def apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock (carryover) transformation.
    
    Formula: adstock[t] = spend[t] + decay * adstock[t-1]
    
    Args:
        spend: Raw weekly spend array
        decay: Retention rate between 0 and 1
                Higher = slower decay (TV ~0.7), Lower = faster (Search ~0.2)
    
    Returns:
        Array of adstocked (effective) spend values
    """
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def apply_saturation(adstocked: np.ndarray, k: float, alpha: float) -> np.ndarray:
    """
    Apply Hill saturation transformation to capture diminishing returns.
    
    Formula: saturation(x) = x^alpha / (x^alpha + k^alpha)
    
    Args:
        adstocked: Adstocked spend array (will be scaled to 0-1 internally)
        k:         Inflection point — spend level at which 50% of max
                   response is achieved (as fraction of max spend, 0-1)
        alpha:     Slope parameter — controls steepness of the curve
                   Higher alpha = sharper S-curve
    
    Returns:
        Array of saturation-transformed values between 0 and 1
    """
    max_val = adstocked.max()
    if max_val == 0:
        return np.zeros_like(adstocked, dtype=float)
    x_scaled = adstocked / max_val
    return (x_scaled ** alpha) / (x_scaled ** alpha + k ** alpha)


def normalise(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise array to 0-1 range.
    Ensures all channels are on the same scale before regression.
    """
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / (max_val - min_val)


def transform_channel(
    spend: np.ndarray,
    decay: float,
    k: float,
    alpha: float,
    normalise_output: bool = True
) -> np.ndarray:
    """
    Full transformation pipeline for a single channel:
    Raw spend → Adstock → Saturation → (Optional) Normalisation
    
    Args:
        spend:            Raw weekly spend array
        decay:            Adstock decay rate
        k:                Saturation inflection point
        alpha:            Saturation slope
        normalise_output: Whether to min-max normalise the final output
    
    Returns:
        Transformed spend array ready for regression
    """
    adstocked  = apply_adstock(spend, decay)
    saturated  = apply_saturation(adstocked, k, alpha)
    if normalise_output:
        return normalise(saturated)
    return saturated


def transform_all_channels(
    df: pd.DataFrame,
    params: Dict[str, Dict[str, float]],
    normalise_output: bool = True
) -> pd.DataFrame:
    """
    Apply full transformation pipeline to all channels.
    
    Args:
        df:     DataFrame with raw spend columns (tv_spend, digital_spend etc)
        params: Dictionary with decay, k, alpha per channel:
                {
                  'tv':      {'decay': 0.7, 'k': 0.5, 'alpha': 2.0},
                  'digital': {'decay': 0.4, 'k': 0.4, 'alpha': 2.5},
                  ...
                }
    
    Returns:
        DataFrame with transformed columns added (tv_transformed etc)
    """
    df_out = df.copy()
    for channel in CHANNELS:
        spend = df[f'{channel}_spend'].values.astype(float)
        p     = params[channel]
        df_out[f'{channel}_transformed'] = transform_channel(
            spend,
            decay=p['decay'],
            k=p['k'],
            alpha=p['alpha'],
            normalise_output=normalise_output
        )
    return df_out


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add week index and sine/cosine seasonality features as control variables.
    
    These are NOT media variables — they control for natural demand patterns
    so the model doesn't incorrectly attribute seasonal revenue to media spend.
    
    Sine and cosine together capture any phase of the annual cycle:
    - sin alone peaks at week 13 (spring)
    - cos alone peaks at week 0/52 (new year)
    - Together they can represent any seasonal peak through their combination
    """
    df_out = df.copy()
    n      = len(df)

    week_idx              = np.arange(n)
    df_out['week_index']  = week_idx / n                                          # slow trend
    df_out['season_sin']  = np.sin(2 * np.pi * week_idx / 52)                    # annual cycle
    df_out['season_cos']  = np.cos(2 * np.pi * week_idx / 52)                    # phase component

    return df_out


if __name__ == "__main__":
    # Quick test using ground truth parameters
    from data.generate_data import GROUND_TRUTH, CHANNELS

    df = pd.read_csv('data/mmm_data.csv')

    # Build params dict from ground truth
    gt = GROUND_TRUTH
    params = {
        ch: {
            'decay': gt['decay_rates'][ch],
            'k':     gt['saturation_k'][ch],
            'alpha': gt['saturation_alpha'][ch]
        }
        for ch in CHANNELS
    }

    df_transformed = transform_all_channels(df, params)
    df_transformed = add_seasonality_features(df_transformed)

    transformed_cols = [f'{ch}_transformed' for ch in CHANNELS]
    print("Transformed columns (first 5 rows):")
    print(df_transformed[transformed_cols].head())
    print(f"\nAll values between 0 and 1: "
          f"{(df_transformed[transformed_cols].max() <= 1).all() and (df_transformed[transformed_cols].min() >= 0).all()}")
    print(f"\nSeasonality features added: {['week_index', 'season_sin', 'season_cos']}")