import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.transformations import transform_all_channels, add_seasonality_features, CHANNELS

# ── Parameter search grids ────────────────────────────────────────────
DECAY_GRID   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_GRID       = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ALPHA_GRID   = [1.0, 1.5, 2.0, 2.5, 3.0]

CONTROL_COLS = ['week_index', 'season_sin', 'season_cos']


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error — average % prediction error."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_vif(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for all feature columns.
    VIF > 5 = moderate multicollinearity concern
    VIF > 10 = severe multicollinearity
    """
    vif_data = []
    features = df_features.values
    for i, col in enumerate(df_features.columns):
        vif = variance_inflation_factor(features, i)
        vif_data.append({'feature': col, 'VIF': round(vif, 2)})
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def fit_model(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """
    Fit Ridge regression model.
    Ridge adds L2 penalty to handle multicollinearity.
    alpha controls regularisation strength — higher = more penalty.
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2   = r2_score(y, y_pred)
    mape = calculate_mape(y, y_pred)
    return model, r2, mape, y_pred


def optimise_channel(
    channel: str,
    df: pd.DataFrame,
    y: np.ndarray,
    other_channels_transformed: pd.DataFrame,
    control_features: pd.DataFrame
) -> dict:
    """
    Grid search to find best decay, k, alpha for a single channel.
    
    Strategy: fix other channels at their current best params,
    optimise one channel at a time. This reduces search space
    from 9^5 × 6^5 × 5^5 to 9 × 6 × 5 per channel.
    """
    best_r2     = -np.inf
    best_params = {}
    results     = []

    raw_spend = df[f'{channel}_spend'].values.astype(float)

    for decay, k, alpha_sat in product(DECAY_GRID, K_GRID, ALPHA_GRID):
        from src.transformations import transform_channel
        ch_transformed = transform_channel(raw_spend, decay, k, alpha_sat)

        X_parts = [
            other_channels_transformed.values,
            ch_transformed.reshape(-1, 1),
            control_features.values
        ]
        X = np.hstack(X_parts)

        _, r2, mape, _ = fit_model(X, y)
        results.append({'decay': decay, 'k': k, 'alpha': alpha_sat, 'r2': r2, 'mape': mape})

        if r2 > best_r2:
            best_r2     = r2
            best_params = {'decay': decay, 'k': k, 'alpha': alpha_sat, 'r2': r2, 'mape': mape}

    return best_params


def run_optimisation(df: pd.DataFrame, n_iterations: int = 2) -> dict:
    """
    Iterative coordinate descent optimisation.
    
    Approach:
    1. Start with initial parameter estimates (midpoint of each grid)
    2. Optimise one channel at a time while holding others fixed
    3. Repeat for n_iterations until parameters stabilise
    
    This is coordinate descent — a standard optimisation technique
    that works well when parameters are somewhat independent.
    """
    y = df['revenue'].values.astype(float)

    # Initial parameters — midpoint of grids
    current_params = {
        ch: {'decay': 0.5, 'k': 0.5, 'alpha': 2.0}
        for ch in CHANNELS
    }

    print("Starting optimisation...")
    print(f"Grid sizes: decay={len(DECAY_GRID)}, k={len(K_GRID)}, alpha={len(ALPHA_GRID)}")
    print(f"Combinations per channel: {len(DECAY_GRID) * len(K_GRID) * len(ALPHA_GRID)}")
    print(f"Iterations: {n_iterations}\n")

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")

        for channel in CHANNELS:
            # Transform all OTHER channels with current best params
            other_params = {ch: current_params[ch] for ch in CHANNELS if ch != channel}
            other_spend  = {ch: df[f'{ch}_spend'].values.astype(float) for ch in CHANNELS if ch != channel}

            from src.transformations import transform_channel
            other_transformed = pd.DataFrame({
                f'{ch}_transformed': transform_channel(
                    other_spend[ch],
                    other_params[ch]['decay'],
                    other_params[ch]['k'],
                    other_params[ch]['alpha']
                )
                for ch in CHANNELS if ch != channel
            })

            # Add seasonality controls
            df_with_season = add_seasonality_features(df)
            control_features = df_with_season[CONTROL_COLS]

            # Optimise this channel
            best = optimise_channel(
                channel, df, y, other_transformed, control_features
            )
            current_params[channel] = best
            print(f"  {channel:10s} → decay={best['decay']:.1f}, k={best['k']:.1f}, alpha={best['alpha']:.1f} | R²={best['r2']:.4f}")

    return current_params


def fit_final_model(df: pd.DataFrame, best_params: dict):
    """
    Fit the final regression model using optimised parameters.
    Also calculates VIF to check multicollinearity.
    """
    df_transformed = transform_all_channels(df, best_params)
    df_transformed = add_seasonality_features(df_transformed)

    transformed_cols = [f'{ch}_transformed' for ch in CHANNELS]
    feature_cols     = transformed_cols + CONTROL_COLS

    X = df_transformed[feature_cols].values
    y = df_transformed['revenue'].values.astype(float)

    model, r2, mape, y_pred = fit_model(X, y)

    # VIF check
    print("\nVIF Analysis (multicollinearity check):")
    vif_df = calculate_vif(df_transformed[feature_cols])
    print(vif_df.to_string(index=False))
    print("\nVIF interpretation: <3 = good, 3-5 = acceptable, >5 = investigate, >10 = severe")

    print(f"\nFinal model performance:")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return model, r2, mape, y_pred, df_transformed, feature_cols


if __name__ == "__main__":
    df = pd.read_csv('data/mmm_data.csv')

    # Run optimisation
    best_params = run_optimisation(df, n_iterations=2)

    print("\n── Best parameters found ──")
    from data.generate_data import GROUND_TRUTH
    print(f"\n{'Channel':<12} {'Found decay':>12} {'True decay':>12} {'Found k':>10} {'True k':>10}")
    print("-" * 60)
    for ch in CHANNELS:
        found_decay = best_params[ch]['decay']
        true_decay  = GROUND_TRUTH['decay_rates'][ch]
        found_k     = best_params[ch]['k']
        true_k      = GROUND_TRUTH['saturation_k'][ch]
        print(f"{ch:<12} {found_decay:>12.1f} {true_decay:>12.1f} {found_k:>10.1f} {true_k:>10.1f}")

    # Fit final model
    print("\n── Fitting final model ──")
    model, r2, mape, y_pred, df_transformed, feature_cols = fit_final_model(df, best_params)