"""
run_test_harness.py
====================
Runs harness_core.py against every hand-derived case in test_cases.py and
reports PASS/FAIL. Run this any time you change harness_core.py, or point
a different implementation's functions at the same test_cases data to
validate Lyra / WinterStorm2030 / StressPoint / Orion against the same
gold-standard reference dataset.

Usage:
    python run_test_harness.py
"""

import sys
import numpy as np
import pandas as pd

from harness_core import (
    nof_score, domain_score, composite_additive, composite_multiplicative,
    apply_environment, monte_carlo_ranking_sensitivity,
)
from test_cases import (
    STRATUM_I_CASES, STRATUM_II_CASES, STRATUM_III_CASES, STRATUM_IV_CASES,
    ALL_CASES, SCENARIO_BANK, SCENARIO_BETAS, EXPECTED_ORDERINGS,
)

results = []


def check(case_id, description, expected, actual, tol, assumption_flagged, stratum):
    passed = abs(expected - actual) <= tol
    results.append(dict(
        id=case_id, stratum=stratum, description=description,
        expected=expected, actual=actual, diff=actual - expected,
        tolerance=tol, passed=passed, assumption_flagged=assumption_flagged,
    ))
    return passed


def run_stratum_I():
    for c in STRATUM_I_CASES:
        actual, z = nof_score(c["X"], c["w"], c["theta"])
        check(c["id"], c["description"], c["expected"], actual, c["tolerance"],
              c["assumption_flagged"], "I")


def run_stratum_II():
    for c in STRATUM_II_CASES:
        actual = domain_score(c["nof_scores"], c["sub_weights"])
        check(c["id"], c["description"], c["expected"], actual, c["tolerance"],
              c["assumption_flagged"], "II")


def run_stratum_III():
    for c in STRATUM_III_CASES:
        add = composite_additive(c["domain_scores"], c["betas"])
        mult = composite_multiplicative(c["domain_scores"], c["betas"])
        check(c["id"] + "_additive", c["description"], c["expected_additive"], add,
              c["tolerance"], c["assumption_flagged"], "III")
        check(c["id"] + "_multiplicative", c["description"], c["expected_multiplicative"], mult,
              c["tolerance"], c["assumption_flagged"], "III")


def run_stratum_IV():
    for c in STRATUM_IV_CASES:
        actual = apply_environment(c["domain_scores"], c["eoc_coeffs"])
        for i, (exp_i, act_i) in enumerate(zip(c["expected"], actual)):
            check(f'{c["id"]}_dom{i}', c["description"], exp_i, act_i, c["tolerance"],
                  c["assumption_flagged"], "IV")


def run_monotonicity_checks():
    """Cross-case relational checks that aren't single hand-calculated
    numbers, but documented required RELATIONSHIPS between cases."""
    def nof(cid):
        row = next(r for r in results if r["id"] == cid)
        return row["actual"]

    checks = [
        ("I-5 > I-1 (raising the highest-weight indicator must increase NOF)",
         nof("I-5_monotonicity_high_weight_indicator") > nof("I-1_baseline")),
        ("I-6 > I-1 (raising the lowest-weight indicator must also increase NOF)",
         nof("I-6_monotonicity_low_weight_indicator") > nof("I-1_baseline")),
        ("(I-5 - I-1) > (I-6 - I-1)  [equal-delta raise on higher-weight indicator moves score more]",
         (nof("I-5_monotonicity_high_weight_indicator") - nof("I-1_baseline")) >
         (nof("I-6_monotonicity_low_weight_indicator") - nof("I-1_baseline"))),
    ]
    for desc, ok in checks:
        results.append(dict(id="RELATIONAL:" + desc[:40], stratum="I-cross", description=desc,
                             expected=np.nan, actual=np.nan, diff=np.nan, tolerance=np.nan,
                             passed=ok, assumption_flagged=False))


def run_scenario_bank():
    """Level 4 face validation: compute additive+multiplicative+EOC-adjusted
    composites for each designed scenario, then check the ORDERINGS a
    domain expert should agree with."""
    scores = {}
    for s in SCENARIO_BANK:
        adj = apply_environment(s["domains"], s["eoc"])
        add = composite_additive(adj, SCENARIO_BETAS)
        mult = composite_multiplicative(adj, SCENARIO_BETAS)
        scores[s["id"]] = dict(additive=add, multiplicative=mult,
                                assumption_flagged=s.get("assumption_flagged", False))

    for a, op, b in EXPECTED_ORDERINGS:
        ok = (scores[a]["additive"] > scores[b]["additive"]) if op == ">" else (scores[a]["additive"] < scores[b]["additive"])
        results.append(dict(
            id=f"ORDERING:{a}{op}{b}", stratum="Scenario", description=f"{a} {op} {b} (additive composite)",
            expected=np.nan, actual=np.nan, diff=np.nan, tolerance=np.nan,
            passed=ok, assumption_flagged=False,
        ))

    catastrophic = scores["S5_one_catastrophic_weak_domain"]
    gap = catastrophic["additive"] - catastrophic["multiplicative"]
    results.append(dict(
        id="ORDERING:S5_mult_collapse_gap", stratum="Scenario",
        description="One catastrophic weak domain: multiplicative composite must fall well below additive (gap > 0.10)",
        expected=np.nan, actual=gap, diff=np.nan, tolerance=np.nan,
        passed=gap > 0.10, assumption_flagged=False,
    ))
    return scores


def run_monte_carlo_demo():
    """Flagship demo tied directly to the reviewer's own worked example:
    Scenario A=[.9,.9,.3] vs Scenario B=[.7,.7,.7], equal baseline weights.
    Shows how ranking robustness differs by aggregation form."""
    domains_a = [0.9, 0.9, 0.3]
    domains_b = [0.7, 0.7, 0.7]
    baseline_betas = [1/3, 1/3, 1/3]

    mc_additive = monte_carlo_ranking_sensitivity(
        domains_a, domains_b, baseline_betas, aggregation="additive",
        concentration=200, n_trials=5000, seed=42)
    mc_mult = monte_carlo_ranking_sensitivity(
        domains_a, domains_b, baseline_betas, aggregation="multiplicative",
        concentration=200, n_trials=5000, seed=42)

    print("\n=== Monte Carlo weight-sensitivity demo ===")
    print(f"Scenario A domains={domains_a}  vs  Scenario B domains={domains_b}")
    print(f"Baseline weights={baseline_betas}, 5000 Dirichlet-perturbed weight draws (concentration=200)")
    print(f"Additive form:       A beats B in {mc_additive['a_wins_fraction']*100:.1f}% of plausible weightings")
    print(f"Multiplicative form: A beats B in {mc_mult['a_wins_fraction']*100:.1f}% of plausible weightings")
    print("If these two percentages diverge sharply, the choice of aggregation form is")
    print("driving your conclusions more than the underlying weight uncertainty is.")
    return mc_additive, mc_mult


def main():
    run_stratum_I()
    run_stratum_II()
    run_stratum_III()
    run_stratum_IV()
    run_monotonicity_checks()
    scenario_scores = run_scenario_bank()
    mc_additive, mc_mult = run_monte_carlo_demo()

    df = pd.DataFrame(results)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 160)

    print("\n=== Full case-by-case results ===")
    print(df[["id", "stratum", "passed", "expected", "actual", "diff", "assumption_flagged"]].to_string(index=False))

    n_total = len(df)
    n_pass = int(df["passed"].sum())
    n_fail = n_total - n_pass
    n_flagged = int(df["assumption_flagged"].sum())
    n_flagged_pass = int(df[df["assumption_flagged"]]["passed"].sum())

    print(f"\n=== Summary ===")
    print(f"Total checks: {n_total}   Passed: {n_pass}   Failed: {n_fail}")
    print(f"Of which ASSUMPTION-FLAGGED (Stratum II placeholder formula): {n_flagged} "
          f"({n_flagged_pass} passed against the placeholder, {n_flagged - n_flagged_pass} failed)")

    out_path = "auracelle_harness_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")

    if n_fail > 0:
        print("\n*** FAILURES DETECTED — see table above. ***")
        sys.exit(1)
    else:
        print("\nAll checks passed against the hand-derived reference values.")


if __name__ == "__main__":
    main()
