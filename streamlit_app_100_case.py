
"""streamlit_app_100_case.py
Streamlit app for the expanded Auracelle 100-case V&V suite.
Rename to streamlit_app.py or set this as the Streamlit Community Cloud main file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from test_cases import ALL_100_CASES
from run_100_case_harness import run_all, composite_multiplicative_zero_collapse
from harness_core import composite_additive, monte_carlo_ranking_sensitivity

st.set_page_config(page_title="Auracelle 100-Case V&V Harness", layout="wide")
st.title("Auracelle Mathematical Verification & Validation Harness")
st.caption("100-case reference suite for computational verification, robustness/stability checks, and validation-supporting scenario behavior.")

df = run_all()
total = len(df); passed = int(df["passed"].sum()); failed = total - passed
flagged = int(df["assumption_flagged"].sum())

st.info("What this verifies: the implementation reproduces independently specified expected outputs for the disclosed equations and declared placeholder functions. It does not yet constitute empirical validation against observed outcome data Y.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reference cases", total)
c2.metric("Passed", passed)
c3.metric("Failed", failed)
c4.metric("Assumption-flagged", flagged)

if failed:
    st.error("Failures detected against the reference suite.")
else:
    st.success("All reference checks passed.")

tabs = st.tabs(["Full results", "Coverage", "Additive vs Multiplicative", "Monte Carlo", "Caveats"])

with tabs[0]:
    st.dataframe(df[["id","category","test_kind","operation","expected","actual","passed","assumption_flagged","description"]], width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="auracelle_100_case_results.csv", mime="text/csv")

with tabs[1]:
    cat = df.groupby(["category", "test_kind"]).agg(cases=("id","count"), passed=("passed","sum"), flagged=("assumption_flagged","sum")).reset_index()
    st.dataframe(cat, width="stretch", hide_index=True)
    fig, ax = plt.subplots(figsize=(9,4))
    df.groupby("category").size().plot(kind="bar", ax=ax)
    ax.set_ylabel("Cases")
    ax.set_title("100-case coverage by category")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Aggregation theory check")
    domains = st.text_input("Domain scores", "0.9, 0.9, 0.3")
    betas = st.text_input("Beta weights", "0.333, 0.333, 0.334")
    try:
        d = [float(x.strip()) for x in domains.split(",")]
        b = [float(x.strip()) for x in betas.split(",")]
        add = composite_additive(d,b)
        mult = composite_multiplicative_zero_collapse(d,b)
        st.metric("Additive", f"{add:.4f}")
        st.metric("Multiplicative", f"{mult:.4f}")
        st.write(f"Gap = **{add - mult:.4f}**")
    except Exception as e:
        st.warning(e)

with tabs[3]:
    st.subheader("Monte Carlo ranking sensitivity")
    concentration = st.slider("Dirichlet concentration", 5, 500, 200, 5)
    n_trials = st.select_slider("Trials", [500,1000,2000,5000,10000], value=5000)
    A = [0.9,0.9,0.3]; B = [0.7,0.7,0.7]; base = [1/3,1/3,1/3]
    mc_add = monte_carlo_ranking_sensitivity(A,B,base,"additive",concentration,n_trials,seed=42)
    # Custom multiplicative zero-collapse not needed because A/B positive
    mc_mult = monte_carlo_ranking_sensitivity(A,B,base,"multiplicative",concentration,n_trials,seed=42)
    m1,m2 = st.columns(2)
    m1.metric("A beats B — additive", f"{mc_add['a_wins_fraction']*100:.1f}%")
    m2.metric("A beats B — multiplicative", f"{mc_mult['a_wins_fraction']*100:.1f}%")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(mc_add["scores_a"], bins=40, alpha=.6, label="A additive")
    ax.hist(mc_add["scores_b"], bins=40, alpha=.6, label="B additive")
    ax.set_title("Additive distribution")
    ax.legend(); st.pyplot(fig)

with tabs[4]:
    st.warning("Stratum II verifies only the placeholder weighted-mean function. It does not validate the final substantive NOF → Domain equation.")
    st.write("EOC values above 1.0 represent favorable/enhancing operating conditions and should be stated as a modeling assumption.")
    st.write("Empirical validation requires observed outcome data Y, calibration, and out-of-sample testing.")
