"""
harness_core.py
================
Reference implementation of the Auracelle four-stratum scoring architecture,
built ONLY to be checked against hand-calculated expected values in
test_cases.py. This module has no authority of its own — if it disagrees
with a hand-calculated case, the CODE is wrong until proven otherwise.

Strata implemented:
  I   — Narrow Operational Factor (NOF):   NOF = sigma(w^T X - theta)
  II  — Domain (BGC) aggregation:          Domain = aggregate(NOF_1..NOF_n)
  III — Composite (g-GWC), two forms:      additive (weighted sum) and
                                            multiplicative (weighted geometric mean)
  IV  — Environmental modulation (EOC):    AdjustedDomain = Domain * EOC

*** STRATUM II FLAG ***
The source material formally specifies Stratum I (logistic NOF score) and
Stratum III (additive/multiplicative composite) with explicit equations.
It does NOT specify a formal NOF -> Domain aggregation formula anywhere in
what's been shared — only that NOFs "aggregate into" a domain score.
This module implements Stratum II as a WEIGHTED ARITHMETIC MEAN of a
domain's NOF scores (equal-weighted by default). This is a placeholder
assumption, not a disclosed formula. Every test case that depends on it is
tagged ASSUMPTION_FLAGGED = True in test_cases.py, and the harness reports
those separately from cases that check only disclosed equations. Confirm
or replace this function before trusting Stratum II results.
"""

import numpy as np


# ---------------------------------------------------------------------------
# STRATUM I — Narrow Operational Factor
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable logistic function. Avoids overflow for large |z|
    by branching on the sign of z (standard stable-sigmoid trick)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out if out.shape else float(out)


def nof_score(X, w, theta):
    """Stratum I: NOF = sigma(w^T X - theta).
    X, w: equal-length sequences of floats (X each expected in [0,1]).
    theta: scalar threshold.
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if X.shape != w.shape:
        raise ValueError(f"X and w must be the same length: got {X.shape} vs {w.shape}")
    z = float(np.dot(w, X) - theta)
    return float(sigmoid(z)), z


# ---------------------------------------------------------------------------
# STRATUM II — Domain (BGC) aggregation  [ASSUMPTION — see module docstring]
# ---------------------------------------------------------------------------

def domain_score(nof_scores, sub_weights=None):
    """Stratum II (ASSUMED FORM): weighted arithmetic mean of a domain's
    NOF scores. If sub_weights is None, equal-weights each NOF.
    Returns the domain score in [0,1] (guaranteed, since it's a convex
    combination of values already in [0,1])."""
    nof_scores = np.asarray(nof_scores, dtype=np.float64)
    if sub_weights is None:
        sub_weights = np.ones_like(nof_scores) / len(nof_scores)
    else:
        sub_weights = np.asarray(sub_weights, dtype=np.float64)
        if not np.isclose(sub_weights.sum(), 1.0, atol=1e-9):
            raise ValueError(f"sub_weights must sum to 1.0, got {sub_weights.sum()}")
    if nof_scores.shape != sub_weights.shape:
        raise ValueError("nof_scores and sub_weights must be the same length")
    return float(np.dot(sub_weights, nof_scores))


# ---------------------------------------------------------------------------
# STRATUM III — Composite (g-GWC): additive and multiplicative forms
# ---------------------------------------------------------------------------

def composite_additive(domain_scores, betas):
    """Composite = sum_j beta_j * Domain_j, requires sum(betas) ~= 1.0."""
    domain_scores = np.asarray(domain_scores, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    if domain_scores.shape != betas.shape:
        raise ValueError("domain_scores and betas must be the same length")
    if not np.isclose(betas.sum(), 1.0, atol=1e-6):
        raise ValueError(f"betas must sum to 1.0 for the additive form, got {betas.sum()}")
    return float(np.dot(betas, domain_scores))


def composite_multiplicative(domain_scores, betas):
    """Composite = prod_j (Domain_j ^ beta_j), the weighted geometric mean.
    Requires sum(betas) ~= 1.0 and all domain_scores > 0 (undefined / -inf
    in the limit at exactly 0, which is the intended 'collapse' behavior —
    this function raises rather than silently returning 0 for a genuine
    zero, so zero-domain scenarios must be tested explicitly with a small
    epsilon and the behavior documented, not hidden)."""
    domain_scores = np.asarray(domain_scores, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    if domain_scores.shape != betas.shape:
        raise ValueError("domain_scores and betas must be the same length")
    if not np.isclose(betas.sum(), 1.0, atol=1e-6):
        raise ValueError(f"betas must sum to 1.0 for the multiplicative form, got {betas.sum()}")
    if np.any(domain_scores <= 0):
        raise ValueError(
            "multiplicative composite is undefined/degenerate for domain_scores <= 0; "
            "use a small epsilon explicitly if you need to test near-zero collapse"
        )
    return float(np.prod(np.power(domain_scores, betas)))


# ---------------------------------------------------------------------------
# STRATUM IV — Environmental Operating Conditions (EOC) modulation
# ---------------------------------------------------------------------------

def apply_environment(domain_scores, eoc_coeffs):
    """AdjustedDomain_j = Domain_j * EOC_j. EOC_j = 1.0 means no
    physical-environment dependency (unmodulated)."""
    domain_scores = np.asarray(domain_scores, dtype=np.float64)
    eoc_coeffs = np.asarray(eoc_coeffs, dtype=np.float64)
    if domain_scores.shape != eoc_coeffs.shape:
        raise ValueError("domain_scores and eoc_coeffs must be the same length")
    return domain_scores * eoc_coeffs


# ---------------------------------------------------------------------------
# Monte Carlo weight-sensitivity testing
# ---------------------------------------------------------------------------

def perturb_weights_dirichlet(baseline_betas, concentration, n_trials, rng):
    """Generate n_trials plausible weight vectors around baseline_betas,
    each summing to 1.0 exactly, using a Dirichlet distribution centered
    on the baseline. Higher `concentration` => tighter clustering around
    baseline (more confident experts); lower => wider disagreement."""
    baseline_betas = np.asarray(baseline_betas, dtype=np.float64)
    alpha = baseline_betas * concentration
    alpha = np.clip(alpha, 1e-3, None)  # avoid zero/negative alpha
    return rng.dirichlet(alpha, size=n_trials)


def monte_carlo_ranking_sensitivity(domains_a, domains_b, baseline_betas,
                                     aggregation="additive", concentration=200,
                                     n_trials=5000, seed=42):
    """For two scenarios (domain-score vectors domains_a, domains_b) and a
    baseline weight vector, draw n_trials plausible weight vectors and
    report the fraction of trials in which scenario A's composite exceeds
    scenario B's. A fraction near 1.0 (or near 0.0) means the ranking is
    robust to reasonable expert disagreement about weights; a fraction
    near 0.5 means the conclusion is an artifact of the particular
    baseline weights chosen."""
    rng = np.random.default_rng(seed)
    weight_draws = perturb_weights_dirichlet(baseline_betas, concentration, n_trials, rng)
    agg_fn = composite_additive if aggregation == "additive" else composite_multiplicative
    a_wins = 0
    scores_a, scores_b = [], []
    for betas in weight_draws:
        ca = agg_fn(domains_a, betas)
        cb = agg_fn(domains_b, betas)
        scores_a.append(ca)
        scores_b.append(cb)
        if ca > cb:
            a_wins += 1
    return {
        "a_wins_fraction": a_wins / n_trials,
        "scores_a": np.array(scores_a),
        "scores_b": np.array(scores_b),
        "weight_draws": weight_draws,
    }
