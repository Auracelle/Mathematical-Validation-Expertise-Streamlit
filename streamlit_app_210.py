"""
streamlit_app.py
=================
Consolidated Streamlit front end for the Auracelle Mathematical Verification
& Validation Harness.

This replaces the earlier streamlit_app.py and streamlit_app_100_case.py.
It uses the expanded 210-case suite while retaining the original interactive
NOF, scenario, aggregation, and Monte Carlo views.

Run locally:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from harness_core import nof_score, composite_additive, monte_carlo_ranking_sensitivity
from run_test_harness import (
    run_all,
    composite_multiplicative_zero_collapse,
    scenario_table,
    run_relational_checks,
    run_scenario_ordering_checks,
)

st.set_page_config(page_title="Auracelle 210-Case V&V Harness", layout="wide")
st.title("Auracelle Mathematical Verification & Validation Harness")
st.caption(
    "210-case reference suite for computational verification, robustness/stability testing, "
    "and validation-supporting scenario behavior across the Auracelle mathematical architecture."
)

# ---------------------------------------------------------------------------
# Reference suite and summary
# ---------------------------------------------------------------------------
base_df = run_all()
rel_df = run_relational_checks(base_df)
scenario_df = scenario_table()
scenario_checks_df = run_scenario_ordering_checks(scenario_df)

total = len(base_df)
passed = int(base_df["passed"].sum())
failed = total - passed
flagged = int(base_df["assumption_flagged"].sum())

st.info(
    "What this verifies: the implementation reproduces independently specified expected outputs "
    "for the disclosed equations and declared placeholder functions across 210 executable reference cases. "
    "Passing these checks verifies computational correctness and specified mathematical behavior; "
    "it does not yet constitute empirical validation against observed outcome data Y."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reference cases", total)
c2.metric("Passed", passed)
c3.metric("Failed", failed)
c4.metric("Assumption-flagged", flagged)

if failed:
    st.error("Failures detected against the 210-case reference suite.")
else:
    st.success("All 100 reference checks pass against the specified expected values.")

st.warning(
    "Stratum II remains assumption-flagged: these cases verify the current placeholder weighted-mean "
    "implementation only; they do not validate the final substantive NOF → Domain equation."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Full results",
    "Coverage",
    "Stratum I — NOF behavior",
    "Additive vs Multiplicative",
    "Scenario bank",
    "Monte Carlo",
    "Caveats",
])

with tabs[0]:
    st.subheader("210-case reference-suite results")
    display_cols = [
        "id", "category", "test_kind", "operation", "expected", "actual",
        "diff", "passed", "assumption_flagged", "description"
    ]
    st.dataframe(base_df[display_cols], width="stretch", hide_index=True)
    st.download_button(
        "Download 210-case results as CSV",
        base_df.to_csv(index=False),
        file_name="auracelle_100_case_results.csv",
        mime="text/csv",
    )

    if not rel_df.empty or not scenario_checks_df.empty:
        st.subheader("Cross-case and scenario relationship checks")
        extra = pd.concat([rel_df, scenario_checks_df], ignore_index=True, sort=False)
        st.dataframe(extra[["id", "category", "test_kind", "passed", "description"]], width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Coverage by category and test type")
    cat = (
        base_df.groupby(["category", "test_kind"])
        .agg(cases=("id", "count"), passed=("passed", "sum"), flagged=("assumption_flagged", "sum"))
        .reset_index()
    )
    st.dataframe(cat, width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    base_df.groupby("category").size().plot(kind="bar", ax=ax)
    ax.set_ylabel("Cases")
    ax.set_title("210-case coverage by category")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    plt.close(fig)

with tabs[2]:
    st.subheader("NOF = σ(wᵀX − θ) — interactive monotonicity explorer")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        w1 = st.slider("w1 (weight on x1)", 0.0, 1.0, 0.5, 0.05)
        w2 = st.slider("w2", 0.0, 1.0, 0.3, 0.05)
        w3 = st.slider("w3", 0.0, 1.0, 0.2, 0.05)
        theta = st.slider("theta (threshold)", -1.0, 1.0, 0.5, 0.05)
        x2_fixed = st.slider("x2 (held fixed)", 0.0, 1.0, 0.6, 0.05)
        x3_fixed = st.slider("x3 (held fixed)", 0.0, 1.0, 0.4, 0.05)

    with col_b:
        x1_sweep = np.linspace(0, 1, 100)
        nof_sweep = [
            nof_score([x1, x2_fixed, x3_fixed], [w1, w2, w3], theta)[0]
            for x1 in x1_sweep
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x1_sweep, nof_sweep)
        ax.set_xlabel("x1")
        ax.set_ylabel("NOF score")
        ax.set_title("NOF vs. x1")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        nondecreasing = all(b >= a for a, b in zip(nof_sweep, nof_sweep[1:]))
        strictly_increasing = all(b > a for a, b in zip(nof_sweep, nof_sweep[1:])) if w1 > 0 else False
        st.write(f"Monotone non-decreasing across sweep: **{nondecreasing}**")
        st.write(f"Strictly increasing across sweep: **{strictly_increasing}**")

with tabs[3]:
    st.subheader("Aggregation theory check")
    st.caption(
        "Compare compensatory additive aggregation with multiplicative aggregation, "
        "which penalizes weak domains more strongly."
    )
    domains = st.text_input("Domain scores (comma-separated)", "0.9, 0.9, 0.3")
    betas = st.text_input("Beta weights (comma-separated)", "0.333, 0.333, 0.334")
    try:
        d = [float(x.strip()) for x in domains.split(",")]
        b = [float(x.strip()) for x in betas.split(",")]
        if len(d) != len(b):
            raise ValueError("Domain-score and beta-weight vectors must have the same length.")
        if not np.isclose(sum(b), 1.0, atol=1e-6):
            st.warning(f"Beta weights sum to {sum(b):.6f}, not 1.0.")
        add = composite_additive(d, b)
        mult = composite_multiplicative_zero_collapse(d, b)
        m1, m2, m3 = st.columns(3)
        m1.metric("Additive", f"{add:.4f}")
        m2.metric("Multiplicative", f"{mult:.4f}")
        m3.metric("Gap", f"{add - mult:.4f}")
    except Exception as e:
        st.warning(f"Check your inputs: {e}")

with tabs[4]:
    st.subheader("Scenario / face-validation bank")
    if scenario_df.empty:
        st.info("No Scenario Bank cases were found in ALL_100_CASES.")
    else:
        st.dataframe(scenario_df, width="stretch", hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(scenario_df))
        width = 0.35
        ax.bar(x - width / 2, scenario_df["additive"], width, label="Additive")
        ax.bar(x + width / 2, scenario_df["multiplicative"], width, label="Multiplicative")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_df["scenario"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Composite score")
        ax.set_title("Scenario bank: additive vs. multiplicative")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        if not scenario_checks_df.empty:
            st.write("**Expected relationship checks:**")
            for _, row in scenario_checks_df.iterrows():
                st.write(("✅ " if row["passed"] else "❌ ") + row["description"])

with tabs[5]:
    st.subheader("Monte Carlo weight-sensitivity")
    st.caption(
        "Draw plausible beta-weight vectors from a Dirichlet distribution and measure how often "
        "Scenario A outranks Scenario B under each aggregation form."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        da_input = st.text_input("Scenario A domain scores", "0.9, 0.9, 0.3", key="mc_da")
        db_input = st.text_input("Scenario B domain scores", "0.7, 0.7, 0.7", key="mc_db")
    with col_b:
        concentration = st.slider("Dirichlet concentration", 5, 500, 200, 5)
        n_trials = st.select_slider("Trials", [500, 1000, 2000, 5000, 10000], value=5000)

    try:
        A = [float(x.strip()) for x in da_input.split(",")]
        B = [float(x.strip()) for x in db_input.split(",")]
        if len(A) != len(B):
            raise ValueError("Scenario A and B must contain the same number of domains.")
        base = [1 / len(A)] * len(A)

        mc_add = monte_carlo_ranking_sensitivity(
            A, B, base, "additive", concentration, n_trials, seed=42
        )
        mc_mult = monte_carlo_ranking_sensitivity(
            A, B, base, "multiplicative", concentration, n_trials, seed=42
        )

        m1, m2 = st.columns(2)
        m1.metric("A beats B — additive", f"{mc_add['a_wins_fraction'] * 100:.1f}%")
        m2.metric("A beats B — multiplicative", f"{mc_mult['a_wins_fraction'] * 100:.1f}%")

        fig, ax = plt.subplots(figsize=(8, 4))
        combined = np.concatenate([mc_add["scores_a"], mc_add["scores_b"]])
        lo, hi = combined.min(), combined.max()
        if np.isclose(lo, hi):
            lo, hi = lo - 0.01, hi + 0.01
        bins = np.linspace(lo, hi, 40)
        ax.hist(mc_add["scores_a"], bins=bins, alpha=0.6, label="Scenario A")
        ax.hist(mc_add["scores_b"], bins=bins, alpha=0.6, label="Scenario B")
        ax.set_title("Additive score distribution")
        ax.set_xlabel("Composite score")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        gap = abs(mc_add["a_wins_fraction"] - mc_mult["a_wins_fraction"])
        if gap > 0.30:
            st.warning(
                f"The aggregation forms differ by {gap * 100:.0f} percentage points in the "
                "probability that A beats B. Aggregation choice is materially affecting the conclusion."
            )
    except Exception as e:
        st.warning(f"Check your inputs: {e}")

with tabs[6]:
    st.subheader("Interpretation and methodological caveats")
    st.write(
        "**Verified here:** implementation-level agreement with the 100 specified reference outputs; "
        "boundary and numerical-stability behavior; additive/multiplicative aggregation behavior; "
        "environmental modulation behavior; and selected scenario relationships."
    )
    st.write(
        "**Not yet established by this harness:** empirical predictive validity, fitted parameter validity, "
        "construct validity across stakeholder populations, or out-of-sample performance against observed Y."
    )
    st.warning(
        "Stratum II verifies only the placeholder weighted-mean function. It does not validate the final "
        "substantive NOF → Domain equation."
    )
    st.write(
        "EOC values above 1.0 represent favorable/enhancing operating conditions. Treat this as an explicit modeling assumption."
    )
    st.write(
        "Exact-zero multiplicative cases use the documented zero-collapse convention: a positive-weight zero domain collapses the composite to 0; a zero-weight term is ignored."
    )
    st.write(
        "Empirical validation requires observed outcome data Y, calibration or estimation, uncertainty analysis, and out-of-sample testing."
    )
