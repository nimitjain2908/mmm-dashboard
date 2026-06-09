# Marketing Mix Modelling (MMM) Dashboard

An end-to-end MMM pipeline that decomposes revenue by marketing channel, estimates adstock and saturation parameters via grid search optimisation, and serves results through an interactive 4-tab Streamlit dashboard.

---

## What it does

- **Estimates adstock decay rates** per channel (TV, Digital, Search, Email, OOH) using coordinate descent grid search — rather than assuming industry priors
- **Models saturation curves** (Hill function) to capture diminishing returns per channel
- **Decomposes revenue** into baseline and per-channel contributions week by week
- **Optimises budget allocation** using SciPy constrained optimisation to maximise predicted revenue
- **Scenario planning** across budget levels showing Revenue per £ and diminishing returns

---

## Project structure
mmm-dashboard/
├── data/
│   └── generate_data.py      # Synthetic dataset with known ground truth parameters
├── src/
│   ├── transformations.py    # Adstock and Hill saturation functions
│   ├── optimiser.py          # Grid search + coordinate descent parameter estimation
│   ├── model.py              # Final Ridge regression model + ROI calculation
│   ├── decomposition.py      # Revenue decomposition by channel
│   ├── budget.py             # SciPy budget optimisation + scenario planning
│   └── types.py              # MMMModel dataclass
├── app.py                    # 4-tab Streamlit dashboard
└── requirements.txt

---

## Model results

| Metric | Value |
|---|---|
| R² | 0.9461 |
| MAPE | 2.31% |
| Baseline share | 60.3% |
| Top channel by ROI | Email (3.09x) |
| Top channel by contribution | Search (18.5% of revenue) |
| Budget optimisation uplift | ~20% on same spend |

---

## Key technical decisions

**Why coordinate descent instead of full grid search?**
A full grid search across all 5 channels simultaneously would require 270⁵ combinations. Coordinate descent optimises one channel at a time while holding others fixed, reducing the search space to 270 × 5 × n_iterations — computationally feasible while finding reliable parameters.

**Why Ridge regression instead of OLS?**
TV and OOH spend are correlated (brand campaigns run both simultaneously), causing multicollinearity. Ridge adds an L2 penalty that stabilises coefficients under multicollinearity, producing more reliable ROI estimates per channel.

**Why synthetic data with ground truth?**
Baking in known parameters (decay rates, saturation shapes) allows model validation — comparing estimated parameters against true values confirms the pipeline is working correctly. The model recovered most parameters within one grid step of ground truth.

---

## Setup

**1. Clone the repository**
git clone https://github.com/nimitjain2908/mmm-dashboard.git
cd mmm-dashboard

**2. Install dependencies**
pip install pandas numpy scikit-learn scipy plotly streamlit statsmodels joblib

**3. Generate synthetic data**
python data/generate_data.py

**4. Train the model**
python -m src.model

**5. Run the dashboard**
streamlit run app.py

---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| Model Overview | R², MAPE, actual vs predicted chart, contribution summary |
| Revenue Decomposition | Weekly stacked bar by channel, saturation curves |
| Budget Optimiser | Optimal allocation for a given budget, revenue uplift, scenario planning |
| Scenario Planner | Real-time revenue prediction as you adjust channel spend |

---

## Built by

Nimit Jain · [LinkedIn](https://linkedin.com/in/nimitjain2908) · [GitHub](https://github.com/nimitjain2908)
