"""
run_test_harness.py
===================
Consolidated Auracelle Mathematical Verification & Validation harness.

This replaces the earlier run_test_harness.py and run_100_case_harness.py.
It runs the expanded ALL_100_CASES reference suite and retains the useful
cross-case monotonicity, scenario-ordering, and Monte Carlo sensitivity checks.

Required repository files:
    harness_core.py
    test_cases.py   # must define ALL_100_CASES

Usage:
    python run_test_harness.py
"""

import sys
from typing import Any

import numpy as np
import pandas as pd

from harness_core import (
    nof_score,
    domain_score,
    composite_additive,
    composite_multiplicative,
    apply_environment,
    monte_carlo_ranking_sensitivity,
)
from test_cases import ALL_100_CASES


def composite_multiplicative_zero_collapse(domain_scores, betas):
    """Multiplicative composite with an explicit zero-collapse convention.

    A zero-valued domain with positive weight collapses the product to zero.
    A zero-weight domain is ignored. This is used for deterministic edge-case
    verification and documents the convention explicitly.
    """
    prod = 1.0
    for d, b in zip(domain_scores, betas):
        d = float(d)
        b = float(b)
        if abs(b) < 1e-15:
            continue
        if d <= 0.0:
            return 0.0
        prod *= d ** b
    return float(prod)


def actual_for(case: dict[str, Any]):
    """Evaluate one reference case using the implementation under test."""
    op = case["operation"]
    inp = case["inputs"]

    if op == "nof_score":
        return nof_score(inp["X"], inp["w"], inp["theta"])[0]
    if op == "domain_score":
        return domain_score(inp["nof_scores"], inp.get("sub_weights"))
    if op == "composite_additive":
        return composite_additive(inp["domain_scores"], inp["betas"])
    if op == "composite_multiplicative":
        return composite_multiplicative(inp["domain_scores"], inp["betas"])
    if op == "composite_multiplicative_zero_collapse":
        return composite_multiplicative_zero_collapse(
            inp["domain_scores"], inp["betas"]
        )
    if op == "apply_environment":
        return list(apply_environment(inp["domain_scores"], inp["eoc_coeffs"]))
    if op == "scenario_additive":
        adjusted = apply_environment(inp["domain_scores"], inp["eoc_coeffs"])
        return composite_additive(adjusted, inp["betas"])

    raise ValueError(f"Unknown operation: {op}")


def close_enough(expected, actual, tol):
    """Tolerance-aware comparison for scalar or vector outputs."""
    if isinstance(expected, (list, tuple, np.ndarray)):
        if len(expected) != len(actual):
            return False
        return max(abs(float(e) - float(a)) for e, a in zip(expected, actual)) <= tol
    return abs(float(expected) - float(actual)) <= tol


def difference(expected, actual):
    """Return a readable scalar difference; vectors use maximum absolute error."""
    if isinstance(expected, (list, tuple, np.ndarray)):
        if len(expected) != len(actual):
            return np.nan
        return max(abs(float(e) - float(a)) for e, a in zip(expected, actual))
    return float(actual) - float(expected)


def run_all() -> pd.DataFrame:
    """Run all 100 reference cases and return case-by-case results."""
    rows = []
    for c in ALL_100_CASES:
        actual = actual_for(c)
        passed = close_enough(c["expected"], actual, c["tolerance"])
        rows.append(
            {
                **c,
                "actual": actual,
                "diff": difference(c["expected"], actual),
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def run_relational_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Retain the legacy cross-case NOF behavior checks using 100-case IDs."""
    by_id = df.set_index("id")
    checks = []

    required = {"I-04", "I-05", "I-06"}
    if required.issubset(by_id.index):
        baseline = float(by_id.loc["I-04", "actual"])
        high = float(by_id.loc["I-05", "actual"])
        low = float(by_id.loc["I-06", "actual"])
        relationships = [
            ("REL-01", "Raising the highest-weight indicator increases NOF", high > baseline),
            ("REL-02", "Raising the lowest-weight indicator increases NOF", low > baseline),
            (
                "REL-03",
                "Equal increase on the higher-weight indicator moves NOF more",
                (high - baseline) > (low - baseline),
            ),
        ]
        for cid, desc, ok in relationships:
            checks.append(
                {
                    "id": cid,
                    "category": "Cross-case relationships",
                    "stratum": "I-cross",
                    "test_kind": "Verification",
                    "operation": "relational_check",
                    "expected": True,
                    "actual": bool(ok),
                    "diff": np.nan,
                    "tolerance": np.nan,
                    "passed": bool(ok),
                    "assumption_flagged": False,
                    "description": desc,
                    "aggregation": "",
                }
            )
    return pd.DataFrame(checks)


def scenario_table() -> pd.DataFrame:
    """Extract and score the six validation-supporting scenario-bank cases."""
    rows = []
    for c in ALL_100_CASES:
        if c.get("category") != "Scenario Bank":
            continue
        inp = c["inputs"]
        adjusted = list(apply_environment(inp["domain_scores"], inp["eoc_coeffs"]))
        additive = composite_additive(adjusted, inp["betas"])
        try:
            multiplicative = composite_multiplicative_zero_collapse(adjusted, inp["betas"])
        except Exception:
            multiplicative = np.nan
        rows.append(
            {
                "id": c["id"],
                "scenario": c["description"],
                "additive": additive,
                "multiplicative": multiplicative,
                "assumption_flagged": c.get("assumption_flagged", False),
            }
        )
    return pd.DataFrame(rows)


def run_scenario_ordering_checks(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Face-validity relationships retained from the earlier harness."""
    if scenarios.empty:
        return pd.DataFrame()
    s = scenarios.set_index("scenario")
    relationships = []

    def add_check(cid, desc, ok):
        relationships.append(
            {
                "id": cid,
                "category": "Scenario relationships",
                "stratum": "Scenario",
                "test_kind": "Validation-supporting",
                "operation": "scenario_ordering",
                "expected": True,
                "actual": bool(ok),
                "diff": np.nan,
                "tolerance": np.nan,
                "passed": bool(ok),
                "assumption_flagged": False,
                "description": desc,
                "aggregation": "Additive/Multiplicative",
            }
        )

    if {"Resilient", "Fragile"}.issubset(s.index):
        add_check("SC-REL-01", "Resilient scores above Fragile", s.loc["Resilient", "additive"] > s.loc["Fragile", "additive"])
    if {"Resilient", "Hostile Environment"}.issubset(s.index):
        add_check("SC-REL-02", "Hostile environment reduces realized score", s.loc["Resilient", "additive"] > s.loc["Hostile Environment", "additive"])
    if {"Catastrophic Weak Domain"}.issubset(s.index):
        gap = float(s.loc["Catastrophic Weak Domain", "additive"] - s.loc["Catastrophic Weak Domain", "multiplicative"])
        add_check("SC-REL-03", "Catastrophic weak domain is penalized more by multiplicative aggregation (gap > 0.10)", gap > 0.10)

    return pd.DataFrame(relationships)


def run_monte_carlo_demo(concentration=200, n_trials=5000, seed=42):
    """Weight-sensitivity demonstration retained from the original harness."""
    domains_a = [0.9, 0.9, 0.3]
    domains_b = [0.7, 0.7, 0.7]
    baseline_betas = [1 / 3, 1 / 3, 1 / 3]

    mc_add = monte_carlo_ranking_sensitivity(
        domains_a,
        domains_b,
        baseline_betas,
        aggregation="additive",
        concentration=concentration,
        n_trials=n_trials,
        seed=seed,
    )
    mc_mult = monte_carlo_ranking_sensitivity(
        domains_a,
        domains_b,
        baseline_betas,
        aggregation="multiplicative",
        concentration=concentration,
        n_trials=n_trials,
        seed=seed,
    )
    return mc_add, mc_mult


def combined_results() -> pd.DataFrame:
    """Return the 100 reference cases plus legacy relational checks."""
    base = run_all()
    rel = run_relational_checks(base)
    sc = run_scenario_ordering_checks(scenario_table())
    frames = [base]
    if not rel.empty:
        frames.append(rel)
    if not sc.empty:
        frames.append(sc)
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    base = run_all()
    extra = combined_results().iloc[len(base):].copy()
    scenarios = scenario_table()
    mc_add, mc_mult = run_monte_carlo_demo()

    print("\n=== Auracelle 100-case reference suite ===")
    print(base[["id", "category", "test_kind", "passed", "expected", "actual", "assumption_flagged"]].to_string(index=False))

    if not extra.empty:
        print("\n=== Cross-case / scenario relationship checks ===")
        print(extra[["id", "category", "passed", "description"]].to_string(index=False))

    n_total = len(base)
    n_pass = int(base["passed"].sum())
    n_fail = n_total - n_pass
    n_flagged = int(base["assumption_flagged"].sum())

    print("\n=== Reference-suite summary ===")
    print(f"Reference cases: {n_total}   Passed: {n_pass}   Failed: {n_fail}")
    print(f"Assumption-flagged: {n_flagged}")
    print("\nBy category/test kind:")
    print(base.groupby(["category", "test_kind"])["passed"].agg(["count", "sum"]))

    print("\n=== Monte Carlo weight-sensitivity demo ===")
    print(f"Additive:       A beats B in {mc_add['a_wins_fraction'] * 100:.1f}% of plausible weightings")
    print(f"Multiplicative: A beats B in {mc_mult['a_wins_fraction'] * 100:.1f}% of plausible weightings")

    base.to_csv("auracelle_100_case_results.csv", index=False)
    combined_results().to_csv("auracelle_vv_results_with_relationship_checks.csv", index=False)
    scenarios.to_csv("auracelle_scenario_scores.csv", index=False)

    print("\nWrote:")
    print("  auracelle_100_case_results.csv")
    print("  auracelle_vv_results_with_relationship_checks.csv")
    print("  auracelle_scenario_scores.csv")

    if n_fail > 0:
        print("\n*** FAILURES DETECTED IN THE 100-CASE REFERENCE SUITE. ***")
        sys.exit(1)
    print("\nAll 100 reference cases passed against the specified expected values.")


if __name__ == "__main__":
    main()
