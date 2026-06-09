import numpy as np
import pandas as pd
import os

# ── Ground truth parameters (baked in for model validation) ──────────
GROUND_TRUTH = {
    'decay_rates': {
        'tv':      0.70,
        'digital': 0.40,
        'search':  0.20,
        'email':   0.15,
        'ooh':     0.60
    },
    'saturation_k': {
        'tv':      0.70,
        'digital': 0.50,
        'search':  0.40,
        'email':   0.30,
        'ooh':     0.60
    },
    'saturation_alpha': {
        'tv':      2.0,
        'digital': 2.5,
        'search':  3.0,
        'email':   2.0,
        'ooh':     1.5
    },
    'coefficients': {
        'tv':      0.35,
        'digital': 0.25,
        'search':  0.45,
        'email':   0.15,
        'ooh':     0.20
    },
    'baseline':        200000,
    'noise_std':       15000
}

CHANNELS = ['tv', 'digital', 'search', 'email', 'ooh']

# ── Spend ranges per channel (weekly £) ──────────────────────────────
SPEND_RANGES = {
    'tv':      (20000, 150000),
    'digital': (10000, 80000),
    'search':  (8000,  60000),
    'email':   (2000,  15000),
    'ooh':     (5000,  40000)
}


def apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock transformation to a spend array."""
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def apply_saturation(adstocked: np.ndarray, k: float, alpha: float) -> np.ndarray:
    """Apply Hill saturation transformation."""
    x_scaled = adstocked / adstocked.max()
    return (x_scaled ** alpha) / (x_scaled ** alpha + k ** alpha)


def generate_spend(n_weeks: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic weekly spend per channel with campaign patterns."""
    np.random.seed(seed)
    weeks = pd.date_range(start='2021-01-04', periods=n_weeks, freq='W')
    spend_data = {'week': weeks}

    for channel in CHANNELS:
        low, high = SPEND_RANGES[channel]
        base_spend = np.random.uniform(low, high, n_weeks)

        # Add campaign bursts — 4 bursts per year for TV and OOH, 6 for digital/search
        n_bursts = 4 if channel in ['tv', 'ooh'] else 6
        burst_weeks = np.random.choice(n_weeks, n_bursts, replace=False)
        burst_duration = 3 if channel in ['tv', 'ooh'] else 2
        for w in burst_weeks:
            end = min(w + burst_duration, n_weeks)
            base_spend[w:end] *= np.random.uniform(1.5, 2.5)
            base_spend[w:end] = np.clip(base_spend[w:end], low, high * 2)

        # TV and OOH correlated with each other (they run together in brand campaigns)
        if channel == 'ooh':
            tv_spend = spend_data.get('tv')
            if tv_spend is not None:
                base_spend = 0.6 * base_spend + 0.4 * (
                    tv_spend / tv_spend.max() * (high - low) + low
                )

        spend_data[f'{channel}_spend'] = base_spend.astype(int)

    return pd.DataFrame(spend_data)


def generate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Generate revenue using ground truth parameters."""
    np.random.seed(42)
    n_weeks = len(df)
    gt = GROUND_TRUTH

    # Seasonality — sine wave peaking in Q4
    week_idx = np.arange(n_weeks)
    seasonality = 30000 * np.sin(2 * np.pi * week_idx / 52 - np.pi / 2) + 30000

    # Media contributions
    total_media = np.zeros(n_weeks)
    contributions = {}

    for channel in CHANNELS:
        raw_spend = df[f'{channel}_spend'].values.astype(float)

        adstocked = apply_adstock(raw_spend, gt['decay_rates'][channel])
        saturated = apply_saturation(
            adstocked,
            gt['saturation_k'][channel],
            gt['saturation_alpha'][channel]
        )

        contrib = gt['coefficients'][channel] * saturated * 500000
        contributions[f'{channel}_contrib'] = contrib
        total_media += contrib

    # Noise
    noise = np.random.normal(0, gt['noise_std'], n_weeks)

    # Final revenue
    df['revenue'] = (
        gt['baseline'] +
        seasonality +
        total_media +
        noise
    ).astype(int)

    # Store contributions for validation
    for channel in CHANNELS:
        df[f'{channel}_true_contrib'] = contributions[f'{channel}_contrib'].astype(int)

    df['true_baseline'] = gt['baseline'] + seasonality

    return df


def generate_dataset(n_weeks: int = 156, save: bool = True) -> pd.DataFrame:
    """Full pipeline — generate spend and revenue."""
    df = generate_spend(n_weeks)
    df = generate_revenue(df)

    if save:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/mmm_data.csv', index=False)
        print(f"Dataset saved to data/mmm_data.csv")
        print(f"Shape: {df.shape}")
        print(f"\nSpend columns: {[c for c in df.columns if 'spend' in c]}")
        print(f"Revenue range: £{df['revenue'].min():,} — £{df['revenue'].max():,}")
        print(f"\nGround truth parameters saved in GROUND_TRUTH dictionary")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 5 rows:")
    print(df[['week'] + [f'{c}_spend' for c in CHANNELS] + ['revenue']].head())