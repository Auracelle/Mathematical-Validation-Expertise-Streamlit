"""
test_cases.py
=============
20 hand-derived test cases across Strata I-IV, plus a 6-scenario face-
validation bank and one flagship Monte Carlo weight-sensitivity demo.

Every numeric `expected` value below was worked out by hand (shown in
`derivation`) BEFORE harness_core.py was run against it. run_test_harness.py
never generates its own "expected" values — it only checks the code against
these. If a future implementation (Lyra, WinterStorm2030, StressPoint,
Orion, ...) disagrees with a case here, treat the implementation as the
thing under suspicion, not this file — unless you deliberately change the
underlying formula, in which case update the derivation AND the expected
value together, by hand, and note why.

`assumption_flagged=True` marks any case that depends on the Stratum II
(NOF -> Domain) aggregation formula, which is NOT formally disclosed in
the source material and is implemented here only as a placeholder
(equal- or explicit-weighted arithmetic mean). Treat failures in flagged
cases as "assumption needs confirmation," not "bug."
"""

TOLERANCE = 1e-6

STRATUM_I_CASES = [
    dict(
        id="I-1_baseline",
        description="Baseline case from the reviewer's worked example.",
        X=[0.8, 0.6, 0.4], w=[0.5, 0.3, 0.2], theta=0.5,
        derivation=(
            "w.X = .5(.8)+.3(.6)+.2(.4) = .40+.18+.08 = .66\n"
            "z = .66 - .50 = .16\n"
            "NOF = 1/(1+e^-.16) ≈ 0.5399"
        ),
        expected=0.539915, tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="I-2_all_zero_inputs",
        description="Boundary: all indicators at 0.",
        X=[0, 0, 0], w=[0.5, 0.3, 0.2], theta=0.5,
        derivation="w.X = 0; z = 0 - .5 = -.5; NOF = 1/(1+e^.5) ≈ 0.3775",
        expected=0.377541, tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="I-3_all_one_inputs",
        description="Boundary: all indicators at 1 (weights sum to 1.0).",
        X=[1, 1, 1], w=[0.5, 0.3, 0.2], theta=0.5,
        derivation="w.X = .5+.3+.2 = 1.0; z = 1.0-.5 = .5; NOF = 1/(1+e^-.5) ≈ 0.6225",
        expected=0.622459, tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="I-4_theta_zero_midpoint",
        description="theta=0 with X all at 0.5: should equal case I-3's z (both z=.5).",
        X=[0.5, 0.5, 0.5], w=[0.5, 0.3, 0.2], theta=0.0,
        derivation="w.X = .5(1.0) = .5; z = .5-0 = .5; NOF = same as I-3 ≈ 0.6225",
        expected=0.622459, tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="I-5_monotonicity_high_weight_indicator",
        description=(
            "Monotonicity check A: raise x1 (highest weight, .5) from .8->.9 "
            "vs baseline I-1. Score MUST increase, and by more than raising a "
            "lower-weight indicator by the same amount (see I-6)."
        ),
        X=[0.9, 0.6, 0.4], w=[0.5, 0.3, 0.2], theta=0.5,
        derivation="w.X=.45+.18+.08=.71; z=.21; NOF=1/(1+e^-.21) ≈ 0.5523",
        expected=0.552308, tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="I-6_monotonicity_low_weight_indicator",
        description=(
            "Monotonicity check B: raise x3 (lowest weight, .2) from .4->.5 "
            "vs baseline I-1, same delta (+0.1) as I-5. Score must increase, "
            "but LESS than I-5 (Δ=+0.0050 here vs Δ=+0.0124 in I-5)."
        ),
        X=[0.8, 0.6, 0.5], w=[0.5, 0.3, 0.2], theta=0.5,
        derivation="w.X=.40+.18+.10=.68; z=.18; NOF=1/(1+e^-.18) ≈ 0.5449",
        expected=0.544879, tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="I-7_exact_threshold_crossing",
        description="Constructed so w.X exactly equals theta -> NOF must be exactly 0.5.",
        X=[1, 0, 0], w=[0.5, 0.3, 0.2], theta=0.5,
        derivation="w.X = .5(1)=.5; z = .5-.5 = 0; NOF = 1/(1+e^0) = 0.5 exactly",
        expected=0.5, tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="I-17_zero_zero_calibration",
        description="Both inputs and threshold at zero -> exact midpoint, independent invariant of I-7.",
        X=[0, 0, 0], w=[0.5, 0.3, 0.2], theta=0.0,
        derivation="w.X=0; z=0-0=0; NOF=0.5 exactly",
        expected=0.5, tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="I-18_theta_equals_max_possible_score",
        description=(
            "theta set equal to the maximum possible w.X (weights sum to 1, "
            "X all at 1) -> even the best-case input only reaches NOF=0.5. "
            "Use this to catch mis-calibrated thresholds in real deployments."
        ),
        X=[1, 1, 1], w=[0.5, 0.3, 0.2], theta=1.0,
        derivation="w.X=1.0 (max possible); z=1.0-1.0=0; NOF=0.5 exactly",
        expected=0.5, tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="I-19_saturation_positive",
        description=(
            "Deliberately extreme unnormalized weights to test sigmoid overflow "
            "safety. z=30 should saturate NOF to (numerically) 1.0, not error/NaN/overflow."
        ),
        X=[1, 1, 1], w=[10, 10, 10], theta=0.0,
        derivation="w.X=30; z=30; NOF=1/(1+e^-30) ≈ 1 - 9.4e-14, i.e. 1.0 to float precision",
        expected=1.0, tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="I-20_saturation_negative",
        description="Mirror of I-19 at the opposite extreme: z=-30 should saturate NOF to ~0.0.",
        X=[1, 1, 1], w=[-10, -10, -10], theta=0.0,
        derivation="w.X=-30; z=-30; NOF=1/(1+e^30) ≈ 9.4e-14, i.e. 0.0 to float precision",
        expected=0.0, tolerance=1e-9, assumption_flagged=False,
    ),
]

STRATUM_II_CASES = [
    dict(
        id="II-8_equal_weight_domain_mean",
        description="[ASSUMPTION-FLAGGED] Equal-weighted mean of 3 NOF scores into one domain.",
        nof_scores=[0.54, 0.6225, 0.5], sub_weights=None,
        derivation="(0.54 + 0.6225 + 0.5) / 3 = 1.6625 / 3 ≈ 0.5542",
        expected=0.554167, tolerance=1e-4, assumption_flagged=True,
    ),
    dict(
        id="II-9_explicit_weighted_domain_mean",
        description="[ASSUMPTION-FLAGGED] Explicitly-weighted mean, mirroring Stratum I's own weighting pattern.",
        nof_scores=[0.8, 0.6, 0.4], sub_weights=[0.5, 0.3, 0.2],
        derivation="(.5)(.8)+(.3)(.6)+(.2)(.4) = .40+.18+.08 = 0.66 exactly",
        expected=0.66, tolerance=1e-9, assumption_flagged=True,
    ),
]

STRATUM_III_CASES = [
    dict(
        id="III-10_additive_vs_multiplicative_collapse_demo",
        description=(
            "The reviewer's own example: one weak domain (.3) among two strong "
            "ones (.9, .9), equal weights. Demonstrates the two aggregation "
            "forms encode DIFFERENT governance theories, not the same math."
        ),
        domain_scores=[0.9, 0.9, 0.3], betas=[1/3, 1/3, 1/3],
        derivation=(
            "Additive: (.9+.9+.3)/3 = .70\n"
            "Multiplicative: (.9 x .9 x .3)^(1/3) = (.243)^(1/3) ≈ 0.6240\n"
            "Gap ≈ 0.076 — the weak domain hurts the geometric form much more."
        ),
        expected_additive=0.70, expected_multiplicative=0.624025,
        tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="III-11_equal_domains_convergence",
        description="Sanity check: when all domains are equal, additive and multiplicative MUST agree exactly.",
        domain_scores=[0.7, 0.7, 0.7], betas=[1/3, 1/3, 1/3],
        derivation="Additive: .7. Multiplicative: (.343)^(1/3) = .7. They must match exactly.",
        expected_additive=0.7, expected_multiplicative=0.7,
        tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="III-12_near_zero_collapse",
        description=(
            "Stress case: one domain near zero (.01) among two very strong "
            "domains (.99, .99). Demonstrates the multiplicative form's "
            "dramatic collapse behavior vs. the additive form's compensation."
        ),
        domain_scores=[0.99, 0.99, 0.01], betas=[1/3, 1/3, 1/3],
        derivation=(
            "Additive: (.99+.99+.01)/3 = 1.99/3 ≈ 0.6633\n"
            "Multiplicative: (.99 x .99 x .01)^(1/3) = (.009801)^(1/3) ≈ 0.2140\n"
            "Gap ≈ 0.449 — the near-zero domain nearly destroys the geometric composite."
        ),
        expected_additive=0.663333, expected_multiplicative=0.214005,
        tolerance=1e-4, assumption_flagged=False,
    ),
    dict(
        id="III-13_unequal_weights",
        description="Non-equal beta weights, moderate domain spread — smaller but still material divergence.",
        domain_scores=[0.8, 0.6, 0.4], betas=[0.5, 0.3, 0.2],
        derivation=(
            "Additive: .5(.8)+.3(.6)+.2(.4) = .40+.18+.08 = 0.66\n"
            "Multiplicative: .8^.5 x .6^.3 x .4^.2 ≈ .8944 x .8580 x .8325 ≈ 0.6389"
        ),
        expected_additive=0.66, expected_multiplicative=0.638855,
        tolerance=1e-4, assumption_flagged=False,
    ),
]

STRATUM_IV_CASES = [
    dict(
        id="IV-14_half_environment",
        description="Reviewer's own example: a hostile environment (EOC=.5) cuts realized capacity in half.",
        domain_scores=[0.80], eoc_coeffs=[0.50],
        derivation="AdjustedDomain = .80 x .50 = 0.40",
        expected=[0.40], tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="IV-15_no_environment_dependency",
        description="Sanity check: EOC=1.0 (no physical dependency) must leave the domain score unchanged.",
        domain_scores=[0.80], eoc_coeffs=[1.00],
        derivation="AdjustedDomain = .80 x 1.00 = 0.80 (unchanged)",
        expected=[0.80], tolerance=1e-9, assumption_flagged=False,
    ),
    dict(
        id="IV-16_weak_domain_mild_environment_compounding",
        description="An already-weak domain (.3) under only a mild environmental penalty (.9) is further reduced.",
        domain_scores=[0.30], eoc_coeffs=[0.90],
        derivation="AdjustedDomain = .30 x .90 = 0.27",
        expected=[0.27], tolerance=1e-9, assumption_flagged=False,
    ),
]

ALL_CASES = STRATUM_I_CASES + STRATUM_II_CASES + STRATUM_III_CASES + STRATUM_IV_CASES


# ---------------------------------------------------------------------------
# Level 4 — Scenario / Face Validation bank (6 deliberately designed cases)
# ---------------------------------------------------------------------------
# These do NOT have hand-calculated numeric targets (that's the point of
# face validation — a domain expert judges plausibility of the RANKING, not
# a specific number). What IS asserted programmatically is the ORDERING
# domain experts would expect, and internal consistency (e.g. EOC=1.0 must
# reproduce the un-modulated composite exactly).
#
# [ASSUMPTION] "Favorable environment" here uses EOC=1.1 (a mild ENHANCEMENT
# above unity). The source material says EOC can "degrade or enhance," but
# does not disclose an upper bound. Confirm a real cap before using EOC>1
# in production — flagged here rather than silently assumed safe.

SCENARIO_BANK = [
    dict(id="S1_clearly_resilient", domains=[0.85, 0.80, 0.90, 0.85], eoc=[1.0, 1.0, 1.0, 1.0]),
    dict(id="S2_clearly_fragile", domains=[0.25, 0.30, 0.20, 0.25], eoc=[1.0, 1.0, 1.0, 1.0]),
    dict(id="S3_strong_institutions_hostile_environment", domains=[0.85, 0.80, 0.90, 0.85], eoc=[0.4, 0.4, 0.4, 0.4],
         assumption_flagged=False),
    dict(id="S4_weak_institutions_favorable_environment", domains=[0.30, 0.30, 0.30, 0.30], eoc=[1.1, 1.1, 1.1, 1.1],
         assumption_flagged=True, assumption_note="EOC>1.0 (enhancement) is an assumed, unconfirmed upper bound."),
    dict(id="S5_one_catastrophic_weak_domain", domains=[0.85, 0.80, 0.90, 0.10], eoc=[1.0, 1.0, 1.0, 1.0]),
    dict(id="S6_balanced_moderate", domains=[0.55, 0.55, 0.55, 0.55], eoc=[1.0, 1.0, 1.0, 1.0]),
]
SCENARIO_BETAS = [0.25, 0.25, 0.25, 0.25]

# Expected qualitative orderings a domain expert should agree with
# (checked on the EOC-adjusted, additive composite unless noted):
EXPECTED_ORDERINGS = [
    ("S1_clearly_resilient", ">", "S6_balanced_moderate"),
    ("S6_balanced_moderate", ">", "S2_clearly_fragile"),
    ("S1_clearly_resilient", ">", "S3_strong_institutions_hostile_environment"),
    # multiplicative-only expectation: the catastrophic-weak-domain scenario
    # should fall MUCH further under multiplicative aggregation than additive
]
