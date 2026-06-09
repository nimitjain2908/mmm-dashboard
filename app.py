import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.model import load_model
from src.decomposition import (
    decompose_revenue, get_total_contributions,
    get_weekly_decomposition, validate_decomposition
)
from src.budget import optimise_budget, run_scenario_planning, predict_revenue_from_spend
from src.transformations import CHANNELS

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MMM Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Marketing Mix Modelling Dashboard")
st.caption("Built by Nimit Jain · Ridge Regression + Adstock & Saturation Transforms + SciPy Budget Optimisation")
# ── Load model and data ───────────────────────────────────────────────
@st.cache_resource
def load():
    mmm = load_model()
    df  = pd.read_csv('data/mmm_data.csv')
    return mmm, df

mmm, df = load()

# ── Pre-compute decomposition ─────────────────────────────────────────
@st.cache_data
def get_decomposition(_mmm):
    decomp   = decompose_revenue(_mmm)
    summary  = get_total_contributions(decomp)
    weekly   = get_weekly_decomposition(decomp)
    return decomp, summary, weekly

decomp, summary, weekly = get_decomposition(mmm)

# ── Current spend reference ───────────────────────────────────────────
current_spend = {ch: df[f'{ch}_spend'].mean() * 4 for ch in CHANNELS}
total_budget  = sum(current_spend.values())

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Model Overview",
    "Revenue Decomposition",
    "Budget Optimiser",
    "Scenario Planner"
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL OVERVIEW
# ═════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{mmm.r2:.4f}")
    col2.metric("MAPE", f"{mmm.mape:.2f}%")
    col3.metric("Baseline Share",
                f"{summary[summary['Channel']=='Baseline']['Share_%'].values[0]:.1f}%")
    col4.metric("Media Channels", str(len(CHANNELS)))

    st.divider()

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Revenue")
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=decomp['week'], y=decomp['actual_revenue'],
        mode='lines', name='Actual',
        line=dict(color='#2E75B6', width=2)
    ))
    fig_avp.add_trace(go.Scatter(
        x=decomp['week'], y=decomp['predicted_revenue'],
        mode='lines', name='Predicted',
        line=dict(color='#FF6B35', width=2, dash='dash')
    ))
    fig_avp.update_layout(
        xaxis_title='Week',
        yaxis_title='Revenue (£)',
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified'
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    st.divider()

    # Summary table
    st.subheader("Revenue Contribution Summary")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.dataframe(
            summary.style.format({
                'Contribution': '£{:,.0f}',
                'Share_%': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    with col_b:
        fig_pie = px.pie(
            summary,
            values='Contribution',
            names='Channel',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Model parameters
    with st.expander("View optimised transformation parameters"):
        params_data = []
        for ch, p in mmm.best_params.items():
            params_data.append({
                'Channel': ch.title(),
                'Decay Rate': p['decay'],
                'Saturation K': p['k'],
                'Saturation Alpha': p['alpha'],
                'ROI': f"{mmm.channel_roi.get(ch, 0):.2f}x"
            })
        st.dataframe(pd.DataFrame(params_data), hide_index=True, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════
# TAB 2 — REVENUE DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Weekly Revenue Decomposition")
    st.caption("Stacked bars show contribution of each channel to total revenue each week")

    COLOUR_MAP = {
        'Tv':       '#2E75B6',
        'Digital':  '#FF6B35',
        'Search':   '#1D9E75',
        'Email':    '#BA7517',
        'Ooh':      '#9B59B6',
        'Baseline': '#95A5A6'
    }

    # Clip negative contributions to zero for clean stacked bar
    weekly_plot = weekly.copy()
    weekly_plot['Revenue'] = weekly_plot['Revenue'].clip(lower=0)

    fig_decomp = px.bar(
        weekly_plot,
        x='week',
        y='Revenue',
        color='Channel',
        color_discrete_map=COLOUR_MAP,
        labels={'week': 'Week', 'Revenue': 'Revenue (£)'}
    )
    fig_decomp.update_layout(
        barmode='stack',
        xaxis_title='Week',
        yaxis_title='Revenue (£)',
        legend=dict(orientation='h', y=1.05),
        hovermode='x unified',
        bargap=0.1
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    st.divider()

    # Saturation curves
    st.subheader("Saturation Curves")
    st.caption("Shows how each channel's effectiveness changes with spend level")

    spend_range = np.linspace(0, 1, 100)
    fig_sat     = go.Figure()

    colours = ['#2E75B6', '#FF6B35', '#1D9E75', '#BA7517', '#9B59B6']
    for ch, colour in zip(CHANNELS, colours):
        p         = mmm.best_params[ch]
        saturated = (spend_range ** p['alpha']) / (
            spend_range ** p['alpha'] + p['k'] ** p['alpha']
        )
        fig_sat.add_trace(go.Scatter(
            x=spend_range * 100,
            y=saturated,
            mode='lines',
            name=ch.title(),
            line=dict(color=colour, width=2)
        ))

    fig_sat.update_layout(
        xaxis_title='Spend Level (% of max)',
        yaxis_title='Response (0-1)',
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified'
    )
    st.plotly_chart(fig_sat, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════
# TAB 3 — BUDGET OPTIMISER
# ═════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Budget Optimiser")
    st.caption("Find the optimal spend allocation for a given total budget")

    budget_input = st.slider(
        "Total monthly budget (£)",
        min_value=int(total_budget * 0.5),
        max_value=int(total_budget * 2.0),
        value=int(total_budget),
        step=10000,
        format="£%d"
    )

    if st.button("▶ Optimise Budget", type="primary"):
        with st.spinner("Running optimisation..."):
            result = optimise_budget(budget_input, mmm, current_spend)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Revenue",  f"£{result['current_revenue']:,.0f}")
        col2.metric("Optimal Revenue",  f"£{result['optimal_revenue']:,.0f}")
        col3.metric("Revenue Uplift",
                    f"£{result['revenue_uplift']:,.0f}",
                    delta=f"{result['uplift_pct']:.1f}%")

        st.divider()

        # Optimal vs current allocation chart
        alloc_data = []
        for ch in CHANNELS:
            alloc_data.append({
                'Channel':  ch.title(),
                'Type':     'Current',
                'Spend':    current_spend[ch]
            })
            alloc_data.append({
                'Channel':  ch.title(),
                'Type':     'Optimal',
                'Spend':    result['optimal_spend'][ch]
            })

        alloc_df  = pd.DataFrame(alloc_data)
        fig_alloc = px.bar(
            alloc_df,
            x='Channel',
            y='Spend',
            color='Type',
            barmode='group',
            color_discrete_map={'Current': '#95A5A6', 'Optimal': '#2E75B6'},
            labels={'Spend': 'Monthly Spend (£)'}
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.divider()

        # Scenario planning
        st.subheader("Scenario Planning")
        st.caption("How does predicted revenue change across different budget levels?")

        with st.spinner("Running scenarios..."):
            scenarios = run_scenario_planning(mmm, budget_input, current_spend)

        fig_scenario = px.line(
            scenarios,
            x='Budget',
            y='Predicted_Revenue',
            markers=True,
            labels={
                'Budget': 'Total Budget (£)',
                'Predicted_Revenue': 'Predicted Revenue (£)'
            }
        )
        fig_scenario.update_traces(line_color='#2E75B6', line_width=2)
        st.plotly_chart(fig_scenario, use_container_width=True)

        with st.expander("View scenario planning table"):
            st.dataframe(
                scenarios.style.format({
                    'Budget':            '£{:,.0f}',
                    'Predicted_Revenue': '£{:,.0f}',
                    'Revenue_per_£':     '{:.2f}x'
                }),
                hide_index=True,
                use_container_width=True
            )

# ═════════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO PLANNER
# ═════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Scenario Planner")
    st.caption("Adjust spend per channel and see predicted revenue impact in real time")

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown("**Adjust monthly spend per channel:**")
        custom_spend = {}
        for ch in CHANNELS:
            custom_spend[ch] = st.slider(
                f"{ch.title()} spend (£)",
                min_value=0,
                max_value=int(current_spend[ch] * 4),
                value=int(current_spend[ch]),
                step=5000,
                key=f"slider_{ch}"
            )

    with col_result:
        predicted = predict_revenue_from_spend(custom_spend, mmm)
        baseline  = predict_revenue_from_spend(
            {ch: 0 for ch in CHANNELS}, mmm
        )
        media_rev = predicted - baseline
        total_custom_spend = sum(custom_spend.values())

        st.markdown("**Predicted outcome:**")
        st.metric("Predicted Monthly Revenue", f"£{predicted:,.0f}")
        st.metric("Media-Driven Revenue",      f"£{media_rev:,.0f}")
        st.metric("Total Spend",               f"£{total_custom_spend:,.0f}")
        if total_custom_spend > 0:
            st.metric("Blended ROI",
                      f"{predicted / total_custom_spend:.2f}x")

        st.divider()

        spend_df  = pd.DataFrame({
            'Channel': [ch.title() for ch in CHANNELS],
            'Spend':   [custom_spend[ch] for ch in CHANNELS]
        })
        fig_spend = px.bar(
        spend_df,
        x='Channel',
        y='Spend',
        color='Channel',
        color_discrete_sequence=list(COLOUR_MAP.values()),
        labels={'Spend': 'Monthly Spend (£)'}
        )
        fig_spend.update_layout(showlegend=False)
        st.plotly_chart(fig_spend, use_container_width=True)