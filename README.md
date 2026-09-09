# Auracelle Mathematical Verification & Validation Test Harness

A gold-standard, hand-derived reference dataset for the four-stratum scoring
architecture (NOF -> Domain -> Composite -> Environmental modulation), to be
run against every Auracelle implementation: Lyra, WinterStorm2030,
StressPoint, Orion, etc.

## Files

- `harness_core.py` — the reference implementation under test (Stratum I-IV
  math). Numerically stable sigmoid; explicit ValueErrors instead of silent
  wrong answers on malformed inputs.
- `test_cases.py` — 20 hand-derived Stratum I-IV cases (expected values were
  worked out by hand BEFORE the code ran against them), plus a 6-scenario
  face-validation bank and the Monte Carlo demo setup.
- `run_test_harness.py` — the plain-Python entry point. Run this for a
  fast pass/fail report and a CSV export.
- `streamlit_app.py` — interactive browser UI over the same harness_core/
  test_cases logic: live sliders for Stratum I weights/threshold, a custom
  additive-vs-multiplicative calculator, the scenario bank, and an
  adjustable Monte Carlo weight-sensitivity demo.
- `.github/workflows/test-harness.yml` — GitHub Actions CI: runs
  `run_test_harness.py` on every push/PR and fails the build if any
  hand-derived check fails.
- `requirements.txt` — pinned-loose dependencies for both the script and
  the Streamlit app.
- `Auracelle_Verification_Validation_Harness.ipynb` — the same content as a
  notebook, with markdown narration, hand-derivation text inline, and plots
  (Stratum I monotonicity, additive-vs-multiplicative divergence, the
  scenario bank, and the Monte Carlo weight-sensitivity demo).
- `auracelle_harness_results.csv` — output of the last run (regenerated
  every time you run the script, the notebook, or the Streamlit app's
  download button).
- `*.png` — the plots the notebook generates, saved standalone.

## Requirements

```
pip install -r requirements.txt
```

(add `jupyter` and `nbconvert` if you want to run the notebook from the
command line rather than in Jupyter/JupyterLab/VS Code directly)

## Running it

**Script:**
```
python run_test_harness.py
```

**Notebook:** open `Auracelle_Verification_Validation_Harness.ipynb` in
Jupyter/JupyterLab/VS Code and run all cells, or from the command line:
```
jupyter nbconvert --to notebook --execute --inplace Auracelle_Verification_Validation_Harness.ipynb
```

**Streamlit app (locally):**
```
streamlit run streamlit_app.py
```
This opens a browser tab with live sliders and tabs for each stratum —
useful for walking a reviewer through the math interactively rather than
handing them a static report.

## Running it on GitHub

Push this whole folder to a GitHub repo. Two things then work automatically:

1. **CI regression check** — `.github/workflows/test-harness.yml` runs
   `run_test_harness.py` on every push and pull request, and fails the
   build (red X) if any hand-derived case fails. The results CSV is
   attached as a downloadable workflow artifact either way. No setup
   needed beyond pushing the `.github/` folder — GitHub runs it for free
   on public repos (and on private repos within the free Actions minutes).
2. **Anyone can clone and run it** — `git clone`, `pip install -r
   requirements.txt`, then either entry point above. GitHub itself doesn't
   execute Python for you outside of Actions; it's the CI runner and the
   distribution point, not a live app host.

## Deploying the Streamlit app for others to use (free)

1. Push the repo to GitHub (same repo as above works fine).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app."
3. Pick the repo/branch, set **Main file path** to `streamlit_app.py`.
4. Deploy. Streamlit Community Cloud installs `requirements.txt`
   automatically and gives you a public URL
   (`https://<something>.streamlit.app`) — this is what you'd send a
   reviewer instead of a screen-share.

Note the free tier is public by default (anyone with the link can open
it) unless you're on a paid tier with access controls — worth keeping in
mind given the provisional-patent status of the underlying architecture;
don't put real domain-specific weights or data into the deployed app,
only the illustrative test cases already in `test_cases.py`.


## What this does NOT do

- It does not validate the real Stratum II (NOF -> Domain) formula, because
  that formula isn't disclosed anywhere in the source material — only a
  placeholder weighted-mean is implemented, and every case depending on it
  is tagged `assumption_flagged=True` in the results. Confirm the real
  formula and swap it into `harness_core.domain_score()` before trusting
  Stratum II/III/IV results downstream of it.
- It does not perform empirical (Level 6) validation against real outcome
  data Y — there is no Y dataset yet. This harness is Levels 1-5 only
  (equation verification, boundary/monotonicity, structural/aggregation-
  choice testing, scenario face-validation, and Monte Carlo weight
  robustness).

## Using this against another implementation (Lyra, WinterStorm2030, ...)

`test_cases.py` is implementation-agnostic — its `expected` values don't
depend on `harness_core.py`. Write a thin adapter exposing that
implementation's NOF/Domain/Composite/EOC functions with the same call
signatures, point `run_test_harness.py`'s imports at the adapter instead,
and diff the resulting CSV against this one. Any case that passes here but
fails there is either a genuine bug in that implementation, or a sign it
made a different, undocumented choice on an assumption-flagged case.
