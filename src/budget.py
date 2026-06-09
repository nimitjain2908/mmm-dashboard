import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
from src.transformations import transform_channel, CHANNELS

# Minimum spend floor per channel (£ per week, scaled to monthly)
MIN_SPEND_FRACTION = 0.05   # at least 5% of current avg spend
MAX_SPEND_FRACTION = 5.0    # at most 5x current avg spend


def predict_revenue_from_spend(
    spend_dict: Dict[str, float],
    mmm,
    n_weeks: int = 4
) -> float:
    """
    Predict total revenue for a given spend allocation over n_weeks.
    Anchors normalisation to training data maximum so predictions
    respond correctly to different spend levels.
    """
    total_revenue = 0.0
    params        = mmm.best_params
    coeffs        = mmm.coefficients
    df_t          = mmm.df_transformed

    for channel in CHANNELS:
        weekly_spend = spend_dict[channel] / n_weeks
        spend_array  = np.array([weekly_spend] * n_weeks)

        p         = params[channel]
        adstocked = _apply_adstock_simple(spend_array, p['decay'])

        # Get the training data max for this channel's RAW spend
        # to anchor the normalisation correctly
        raw_col      = f'{channel}_spend'
        df_raw       = pd.read_csv('data/mmm_data.csv')
        train_max    = _apply_adstock_simple(
            df_raw[raw_col].values.astype(float), p['decay']
        ).max()

        if train_max == 0:
            saturated = np.zeros(n_weeks)
        else:
            x_scaled  = adstocked / train_max
            saturated = (x_scaled ** p['alpha']) / (
                x_scaled ** p['alpha'] + p['k'] ** p['alpha']
            )

        coef           = coeffs.get(f'{channel}_transformed', 0)
        total_revenue += coef * saturated.sum()

    # Baseline
    avg_controls = df_t[['week_index', 'season_sin', 'season_cos']].mean()
    baseline_rev = mmm.model.intercept_
    for col in ['week_index', 'season_sin', 'season_cos']:
        baseline_rev += coeffs.get(col, 0) * avg_controls[col]

    total_revenue += baseline_rev * n_weeks
    return total_revenue


def _apply_adstock_simple(spend: np.ndarray, decay: float) -> np.ndarray:
    """Lightweight adstock for optimisation — same formula as transformations.py."""
    adstocked    = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def optimise_budget(
    total_budget: float,
    mmm,
    current_spend: Dict[str, float] = None,
    n_weeks: int = 4
) -> Dict:
    """
    Find the optimal spend allocation that maximises predicted revenue
    for a given total budget.

    Args:
        total_budget:  Total budget to allocate across all channels (£)
        mmm:           Fitted MMMModel object
        current_spend: Current/historical avg monthly spend per channel
                       Used to set bounds. If None, uses equal split.
        n_weeks:       Planning horizon in weeks (default 4 = 1 month)

    Returns:
        Dictionary with optimal allocation and predicted revenue
    """
    n_channels = len(CHANNELS)

    # Default current spend — equal split
    if current_spend is None:
        current_spend = {ch: total_budget / n_channels for ch in CHANNELS}

    # Bounds — each channel between 10% and 300% of current spend
    bounds = []
    for ch in CHANNELS:
        min_s = current_spend[ch] * MIN_SPEND_FRACTION
        max_s = current_spend[ch] * MAX_SPEND_FRACTION
        bounds.append((min_s, max_s))

    # Constraint — total spend must equal budget
    constraints = [{
        'type': 'eq',
        'fun':  lambda x: x.sum() - total_budget
    }]

    # Objective — negative revenue (we minimise, so this maximises revenue)
    def objective(spend_array: np.ndarray) -> float:
        spend_dict = dict(zip(CHANNELS, spend_array))
        return -predict_revenue_from_spend(spend_dict, mmm, n_weeks)

    # Initial guess — proportional to current spend, scaled to budget
    current_total = sum(current_spend.values())
    x0 = np.array([
        current_spend[ch] / current_total * total_budget
        for ch in CHANNELS
    ])

    # Run optimisation
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    optimal_spend = dict(zip(CHANNELS, result.x))
    optimal_revenue = predict_revenue_from_spend(optimal_spend, mmm, n_weeks)
    current_revenue = predict_revenue_from_spend(current_spend, mmm, n_weeks)

    return {
        'optimal_spend':    optimal_spend,
        'optimal_revenue':  optimal_revenue,
        'current_revenue':  current_revenue,
        'revenue_uplift':   optimal_revenue - current_revenue,
        'uplift_pct':       (optimal_revenue - current_revenue) / current_revenue * 100,
        'success':          result.success,
        'total_budget':     total_budget
    }


def run_scenario_planning(
    mmm,
    base_budget:   float,
    current_spend: Dict[str, float] = None,
    n_scenarios:   int = 9
) -> pd.DataFrame:
    """
    Run optimisation across a range of budget levels for scenario planning.
    Shows how predicted revenue changes as total budget increases or decreases.

    Budget range: 50% to 150% of base budget in equal steps.
    """
    budgets  = np.linspace(base_budget * 0.5, base_budget * 1.5, n_scenarios)
    scenarios = []

    for budget in budgets:
        result = optimise_budget(budget, mmm, current_spend)
        scenarios.append({
            'Budget':           round(budget),
            'Predicted_Revenue': round(result['optimal_revenue']),
            'Revenue_per_£':    round(result['optimal_revenue'] / budget, 2),
            **{f'{ch}_spend': round(result['optimal_spend'][ch])
               for ch in CHANNELS}
        })

    return pd.DataFrame(scenarios)


if __name__ == "__main__":
    from src.model import load_model

    mmm = load_model()

    # Use average monthly spend from data as current spend reference
    df  = pd.read_csv('data/mmm_data.csv')
    current_spend = {
        ch: df[f'{ch}_spend'].mean() * 4
        for ch in CHANNELS
    }

    total_budget = sum(current_spend.values())
    print(f"Total monthly budget: £{total_budget:,.0f}")
    print(f"\nCurrent spend allocation:")
    for ch, spend in current_spend.items():
        print(f"  {ch:<12} £{spend:>10,.0f}")

    print(f"\nRunning budget optimisation...")
    result = optimise_budget(total_budget, mmm, current_spend)

    print(f"\nOptimal spend allocation:")
    for ch, spend in result['optimal_spend'].items():
        change = spend - current_spend[ch]
        arrow  = "▲" if change > 0 else "▼"
        print(f"  {ch:<12} £{spend:>10,.0f}  {arrow} £{abs(change):,.0f}")

    print(f"\nRevenue impact:")
    print(f"  Current predicted:  £{result['current_revenue']:>12,.0f}")
    print(f"  Optimal predicted:  £{result['optimal_revenue']:>12,.0f}")
    print(f"  Uplift:             £{result['revenue_uplift']:>12,.0f} ({result['uplift_pct']:.1f}%)")

    print(f"\nRunning scenario planning...")
    scenarios = run_scenario_planning(mmm, total_budget, current_spend)
    print(scenarios[['Budget', 'Predicted_Revenue', 'Revenue_per_£']].to_string(index=False))