"""
streamlit_app.py
=================
Interactive Streamlit front end for the Auracelle Verification & Validation
harness. Same underlying math as run_test_harness.py / the notebook —
harness_core.py and test_cases.py are unchanged and shared across all three.

Run locally:
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud:
    1. Push this whole folder to a GitHub repo (public or private).
    2. Go to share.streamlit.io -> "New app" -> pick the repo/branch ->
       set "Main file path" to streamlit_app.py.
    3. It installs requirements.txt automatically and gives you a public URL.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from harness_core import (
    nof_score, domain_score, composite_additive, composite_multiplicative,
    apply_environment, monte_carlo_ranking_sensitivity,
)
from test_cases import (
    STRATUM_I_CASES, STRATUM_II_CASES, STRATUM_III_CASES, STRATUM_IV_CASES,
    SCENARIO_BANK, SCENARIO_BETAS, EXPECTED_ORDERINGS,
)

st.set_page_config(page_title="Auracelle V&V Harness", layout="wide")
st.title("Auracelle Mathematical Verification & Validation Harness")
st.caption(
    "Gold-standard hand-derived reference dataset for Strata I-IV. "
    "Run this against Lyra, WinterStorm2030, StressPoint, Orion, etc. "
    "by swapping the harness_core import for an adapter to that implementation."
)

# ---------------------------------------------------------------------------
# Run all hand-derived checks
# ---------------------------------------------------------------------------

def run_all_checks():
    rows = []

    def check(case_id, description, expected, actual, tol, flagged, stratum):
        passed = abs(expected - actual) <= tol
        rows.append(dict(id=case_id, stratum=stratum, description=description,
                          expected=expected, actual=actual, diff=actual - expected,
                          tolerance=tol, passed=passed, assumption_flagged=flagged))

    for c in STRATUM_I_CASES:
        actual, _ = nof_score(c["X"], c["w"], c["theta"])
        check(c["id"], c["description"], c["expected"], actual, c["tolerance"], c["assumption_flagged"], "I")

    for c in STRATUM_II_CASES:
        actual = domain_score(c["nof_scores"], c["sub_weights"])
        check(c["id"], c["description"], c["expected"], actual, c["tolerance"], c["assumption_flagged"], "II")

    for c in STRATUM_III_CASES:
        add = composite_additive(c["domain_scores"], c["betas"])
        mult = composite_multiplicative(c["domain_scores"], c["betas"])
        check(c["id"] + "_additive", c["description"], c["expected_additive"], add,
              c["tolerance"], c["assumption_flagged"], "III")
        check(c["id"] + "_multiplicative", c["description"], c["expected_multiplicative"], mult,
              c["tolerance"], c["assumption_flagged"], "III")

    for c in STRATUM_IV_CASES:
        actual = apply_environment(c["domain_scores"], c["eoc_coeffs"])
        for i, (exp_i, act_i) in enumerate(zip(c["expected"], actual)):
            check(f'{c["id"]}_dom{i}', c["description"], exp_i, act_i, c["tolerance"], c["assumption_flagged"], "IV")

    return pd.DataFrame(rows)


df = run_all_checks()
n_total, n_pass = len(df), int(df["passed"].sum())
n_flagged = int(df["assumption_flagged"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total checks", n_total)
c2.metric("Passed", n_pass, delta=f"{n_pass - n_total}" if n_pass < n_total else None)
c3.metric("Failed", n_total - n_pass)
c4.metric("Assumption-flagged (Stratum II)", n_flagged)

if n_pass < n_total:
    st.error("Failures detected against the hand-derived reference values — see table below.")
else:
    st.success("All checks pass against the hand-derived reference values.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Full results", "Stratum I — NOF behavior", "Stratum III — Additive vs. Multiplicative",
    "Scenario bank", "Monte Carlo sensitivity",
])

with tab1:
    st.dataframe(df, width='stretch', hide_index=True)
    st.download_button("Download results as CSV", df.to_csv(index=False),
                        file_name="auracelle_harness_results.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Stratum I — interactive monotonicity explorer
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("NOF = sigma(w^T X - theta) — sweep one indicator")
    colA, colB = st.columns([1, 2])
    with colA:
        w1 = st.slider("w1 (weight on x1)", 0.0, 1.0, 0.5, 0.05)
        w2 = st.slider("w2", 0.0, 1.0, 0.3, 0.05)
        w3 = st.slider("w3", 0.0, 1.0, 0.2, 0.05)
        theta = st.slider("theta (threshold)", -1.0, 1.0, 0.5, 0.05)
        x2_fixed = st.slider("x2 (held fixed)", 0.0, 1.0, 0.6, 0.05)
        x3_fixed = st.slider("x3 (held fixed)", 0.0, 1.0, 0.4, 0.05)
    with colB:
        x1_sweep = np.linspace(0, 1, 100)
        nof_sweep = [nof_score([x1, x2_fixed, x3_fixed], [w1, w2, w3], theta)[0] for x1 in x1_sweep]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x1_sweep, nof_sweep)
        ax.set_xlabel("x1"); ax.set_ylabel("NOF score")
        ax.set_title("NOF vs. x1 (should be monotonically increasing)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        is_monotonic = all(b >= a for a, b in zip(nof_sweep, nof_sweep[1:]))
        st.write(f"Strictly monotonic across sweep: **{is_monotonic}**")

# ---------------------------------------------------------------------------
# Stratum III — additive vs multiplicative
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Additive vs. multiplicative aggregation across test cases")
    rows = []
    for c in STRATUM_III_CASES:
        add = composite_additive(c["domain_scores"], c["betas"])
        mult = composite_multiplicative(c["domain_scores"], c["betas"])
        rows.append(dict(case=c["id"], domains=str(c["domain_scores"]), additive=add,
                          multiplicative=mult, gap=add - mult))
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, width='stretch', hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width / 2, comp_df["additive"], width, label="Additive")
    ax.bar(x + width / 2, comp_df["multiplicative"], width, label="Multiplicative")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("III-", "") for c in comp_df["case"]], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Composite score")
    ax.legend()
    st.pyplot(fig)

    st.info(
        "Try it yourself below: two custom domain vectors and a shared weight vector, "
        "to see how much the two aggregation forms disagree on YOUR numbers."
    )
    d_input = st.text_input("Domain scores (comma-separated, in [0,1])", "0.9, 0.9, 0.3")
    b_input = st.text_input("Beta weights (comma-separated, must sum to 1.0)", "0.333, 0.333, 0.334")
    try:
        d_vals = [float(x.strip()) for x in d_input.split(",")]
        b_vals = [float(x.strip()) for x in b_input.split(",")]
        add_custom = composite_additive(d_vals, b_vals)
        mult_custom = composite_multiplicative(d_vals, b_vals)
        st.write(f"Additive = **{add_custom:.4f}**  |  Multiplicative = **{mult_custom:.4f}**  |  Gap = **{add_custom - mult_custom:.4f}**")
    except Exception as e:
        st.warning(f"Check your inputs: {e}")

# ---------------------------------------------------------------------------
# Scenario bank
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Level 4 — Scenario / face-validation bank")
    scenario_rows = []
    scenario_scores = {}
    for s in SCENARIO_BANK:
        adj = apply_environment(s["domains"], s["eoc"])
        add = composite_additive(adj, SCENARIO_BETAS)
        mult = composite_multiplicative(adj, SCENARIO_BETAS)
        scenario_scores[s["id"]] = dict(additive=add, multiplicative=mult)
        scenario_rows.append(dict(scenario=s["id"], additive=add, multiplicative=mult,
                                   assumption_flagged=s.get("assumption_flagged", False)))
    st.dataframe(pd.DataFrame(scenario_rows), width='stretch', hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [s["id"].replace("_", "\n") for s in SCENARIO_BANK]
    adds = [scenario_scores[s["id"]]["additive"] for s in SCENARIO_BANK]
    mults = [scenario_scores[s["id"]]["multiplicative"] for s in SCENARIO_BANK]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, adds, width, label="Additive")
    ax.bar(x + width / 2, mults, width, label="Multiplicative")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Composite score")
    ax.legend()
    st.pyplot(fig)

    st.write("**Expected orderings check:**")
    for a, op, b in EXPECTED_ORDERINGS:
        ok = (scenario_scores[a]["additive"] > scenario_scores[b]["additive"]) if op == ">" \
            else (scenario_scores[a]["additive"] < scenario_scores[b]["additive"])
        st.write(("✅ " if ok else "❌ ") + f"{a} {op} {b}")

# ---------------------------------------------------------------------------
# Monte Carlo sensitivity — interactive
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Monte Carlo weight-sensitivity")
    st.caption(
        "Draw plausible weight vectors around a baseline (Dirichlet distribution) and see "
        "how often Scenario A beats Scenario B under each aggregation form."
    )
    colA, colB = st.columns(2)
    with colA:
        da_input = st.text_input("Scenario A domain scores", "0.9, 0.9, 0.3", key="mc_da")
        db_input = st.text_input("Scenario B domain scores", "0.7, 0.7, 0.7", key="mc_db")
    with colB:
        concentration = st.slider("Dirichlet concentration (higher = experts agree more)", 5, 500, 200, 5)
        n_trials = st.select_slider("Number of trials", options=[500, 1000, 2000, 5000, 10000], value=5000)

    try:
        domains_a = [float(x.strip()) for x in da_input.split(",")]
        domains_b = [float(x.strip()) for x in db_input.split(",")]
        baseline_betas = [1 / len(domains_a)] * len(domains_a)

        mc_add = monte_carlo_ranking_sensitivity(domains_a, domains_b, baseline_betas,
                                                  aggregation="additive", concentration=concentration,
                                                  n_trials=n_trials, seed=42)
        mc_mult = monte_carlo_ranking_sensitivity(domains_a, domains_b, baseline_betas,
                                                   aggregation="multiplicative", concentration=concentration,
                                                   n_trials=n_trials, seed=42)

        m1, m2 = st.columns(2)
        m1.metric("A beats B (additive)", f"{mc_add['a_wins_fraction']*100:.1f}%")
        m2.metric("A beats B (multiplicative)", f"{mc_mult['a_wins_fraction']*100:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, mc, title in [(axes[0], mc_add, "Additive"), (axes[1], mc_mult, "Multiplicative")]:
            combined = np.concatenate([mc["scores_a"], mc["scores_b"]])
            lo, hi = combined.min(), combined.max()
            if np.isclose(lo, hi):
                lo, hi = lo - 0.01, hi + 0.01
            bin_edges = np.linspace(lo, hi, 40)
            ax.hist(mc["scores_a"], bins=bin_edges, alpha=0.6, label="Scenario A")
            ax.hist(mc["scores_b"], bins=bin_edges, alpha=0.6, label="Scenario B")
            ax.set_title(f"{title}")
            ax.set_xlabel("Composite score")
            ax.legend()
        st.pyplot(fig)

        gap = abs(mc_add["a_wins_fraction"] - mc_mult["a_wins_fraction"])
        if gap > 0.3:
            st.warning(
                f"The two aggregation forms disagree sharply on how often A beats B "
                f"(gap = {gap*100:.0f} percentage points). The choice of aggregation form is "
                f"driving this conclusion more than weight uncertainty is."
            )
    except Exception as e:
        st.warning(f"Check your inputs: {e}")

st.divider()
st.caption(
    "Stratum II (NOF -> Domain) uses a placeholder weighted-mean formula — not a disclosed "
    "equation. Confirm the real formula before trusting Stratum II/III/IV results downstream."
)
