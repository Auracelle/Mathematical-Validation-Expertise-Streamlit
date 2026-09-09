
"""run_100_case_harness.py
Run the Auracelle 100-case mathematical verification suite.
"""
import math
import pandas as pd
from test_cases import ALL_100_CASES
from harness_core import nof_score, domain_score, composite_additive, composite_multiplicative, apply_environment


def composite_multiplicative_zero_collapse(domain_scores, betas):
    """Documented zero-collapse convention for edge-case testing.
    Existing harness_core may raise on exact zero; this wrapper treats a positive-weight
    zero domain as a deterministic collapse to 0.0 and ignores zero-beta terms.
    """
    prod = 1.0
    for d, b in zip(domain_scores, betas):
        if abs(b) < 1e-15:
            continue
        if d <= 0:
            return 0.0
        prod *= d ** b
    return float(prod)


def actual_for(case):
    op, inp = case["operation"], case["inputs"]
    if op == "nof_score":
        return nof_score(inp["X"], inp["w"], inp["theta"])[0]
    if op == "domain_score":
        return domain_score(inp["nof_scores"], inp.get("sub_weights"))
    if op == "composite_additive":
        return composite_additive(inp["domain_scores"], inp["betas"])
    if op == "composite_multiplicative_zero_collapse":
        return composite_multiplicative_zero_collapse(inp["domain_scores"], inp["betas"])
    if op == "apply_environment":
        return list(apply_environment(inp["domain_scores"], inp["eoc_coeffs"]))
    if op == "scenario_additive":
        adjusted = apply_environment(inp["domain_scores"], inp["eoc_coeffs"])
        return composite_additive(adjusted, inp["betas"])
    raise ValueError(f"Unknown operation: {op}")


def close_enough(expected, actual, tol):
    if isinstance(expected, list):
        return max(abs(float(e)-float(a)) for e, a in zip(expected, actual)) <= tol
    return abs(float(expected) - float(actual)) <= tol


def run_all():
    rows = []
    for c in ALL_100_CASES:
        actual = actual_for(c)
        passed = close_enough(c["expected"], actual, c["tolerance"])
        rows.append({**c, "actual": actual, "passed": passed})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = run_all()
    df.to_csv("auracelle_100_case_results.csv", index=False)
    print(f"Passed {df['passed'].sum()} / {len(df)} checks")
    print(df.groupby(['category', 'test_kind'])['passed'].agg(['count', 'sum']))
