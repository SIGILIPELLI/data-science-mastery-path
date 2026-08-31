# 02 · Advanced Causal Inference

Randomized experiments (Level 2) aren't always possible — you can't
randomize who gets a minimum wage increase, a chronic illness, or a
decade of tenure at a company. This module covers the main
quasi-experimental methods used to estimate causal effects from
observational data: difference-in-differences, instrumental variables,
regression discontinuity, and synthetic control.

## Difference-in-differences (DiD)

Used when a treatment rolls out to one group but not another, at a known
point in time, and you have pre/post data for both.

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

rng = np.random.default_rng(0)
n_per_group = 200

df = pd.concat([
    pd.DataFrame({
        "unit": range(n_per_group),
        "group": "treated",
        "period": "pre",
        "outcome": rng.normal(50, 5, n_per_group),
    }),
    pd.DataFrame({
        "unit": range(n_per_group, 2 * n_per_group),
        "group": "control",
        "period": "pre",
        "outcome": rng.normal(48, 5, n_per_group),
    }),
    pd.DataFrame({
        "unit": range(n_per_group),
        "group": "treated",
        "period": "post",
        "outcome": rng.normal(58, 5, n_per_group),   # +8 baseline drift +treatment
    }),
    pd.DataFrame({
        "unit": range(n_per_group, 2 * n_per_group),
        "group": "control",
        "period": "post",
        "outcome": rng.normal(51, 5, n_per_group),    # +3 baseline drift only
    }),
])

df["is_treated"] = (df["group"] == "treated").astype(int)
df["is_post"] = (df["period"] == "post").astype(int)

model = smf.ols("outcome ~ is_treated * is_post", data=df).fit()
print(model.params["is_treated:is_post"])
```

```text
5.03
```

The `is_treated:is_post` coefficient is the DiD estimate: it nets out
both groups' baseline post-period drift (the "difference" over time,
"differenced" between groups), isolating the roughly +5 effect that's
attributable to treatment rather than to a trend both groups would have
seen anyway.

**Key assumption — parallel trends**: absent treatment, the two groups
would have moved in parallel. This isn't testable directly on post-period
data, but is commonly checked by plotting multiple *pre*-period points and
confirming the groups track each other before treatment starts.

## Instrumental variables (IV)

Used when treatment assignment is confounded, but there's a variable (the
**instrument**) that affects treatment without directly affecting the
outcome except through treatment.

```text
Instrument Z → Treatment X → Outcome Y
                   ↑
            Confounder U (affects both X and Y, unobserved)

Valid instrument Z must satisfy:
  1. Relevance: Z is correlated with X
  2. Exclusion: Z affects Y ONLY through X (no direct path Z → Y)
```

```python
# Two-stage least squares (2SLS), by hand
# Stage 1: regress treatment on instrument
stage1 = smf.ols("X ~ Z", data=iv_df).fit()
iv_df["X_hat"] = stage1.predict(iv_df)

# Stage 2: regress outcome on the PREDICTED (instrument-driven) treatment
stage2 = smf.ols("Y ~ X_hat", data=iv_df).fit()
print(stage2.params["X_hat"])   # consistent estimate of the causal effect of X on Y
```

The exclusion restriction can't be tested statistically — it's an
assumption defended with domain argument (e.g. "distance to the nearest
college affects whether someone attends college, but doesn't independently
affect their wages except through education"). A weak or implausible
exclusion argument makes an IV estimate no more trustworthy than the
naive regression it was meant to fix.

## Regression discontinuity (RD)

Used when treatment is assigned by a hard cutoff on a running variable
(a test score threshold for a scholarship, an age cutoff for a policy).

```python
df["running_var"] = rng.normal(0, 10, 400)
cutoff = 0
df["treated"] = (df["running_var"] >= cutoff).astype(int)
df["outcome"] = 20 + 0.5 * df["running_var"] + 4 * df["treated"] + rng.normal(0, 3, 400)

# Local linear regression within a bandwidth around the cutoff
bw = 15
local = df[df["running_var"].between(-bw, bw)].copy()
rd_model = smf.ols("outcome ~ running_var * treated", data=local).fit()
print(rd_model.params["treated"])   # ~ discontinuity at the cutoff = causal effect
```

The identifying idea: units just above and just below the cutoff are
nearly identical in every other respect, so any jump in outcome exactly at
the cutoff is attributable to treatment. Sensitivity to bandwidth choice
(`bw` above) is a standard robustness check — report the estimate across
a few bandwidths, not just one.

## Synthetic control

Used for a single treated unit (one state, one company, one market) with
many untreated candidates, no natural instrument, and no sharp cutoff.

```python
# Weighted combination of control units that best matches
# the treated unit's PRE-treatment outcome trajectory
from scipy.optimize import nnls

pre_treated = np.array([10, 11, 12, 13, 14])          # treated unit, pre-period
pre_controls = np.column_stack([                        # candidate donor units, pre-period
    [9, 10, 11, 12, 13],
    [12, 12, 13, 14, 14],
    [8, 9, 9, 10, 11],
])

weights, _ = nnls(pre_controls, pre_treated)
weights = weights / weights.sum()
print(weights)   # non-negative weights summing to 1 — the "synthetic" twin
```

Once weights are fit on the pre-period, the same weights construct a
synthetic counterfactual for the *post*-period; the gap between the real
treated unit and its synthetic twin post-treatment is the effect estimate.
The pre-period fit quality is the main diagnostic — a synthetic control
that doesn't track the real unit closely before treatment isn't credible
as a counterfactual after it.

## Choosing a method

| Situation | Method |
|---|---|
| Treatment rolls out to a group at a known time; have pre/post for a control group | Difference-in-differences |
| Confounded assignment, but a plausible external "nudge" exists | Instrumental variables |
| Treatment assigned by a hard cutoff on a running variable | Regression discontinuity |
| One treated unit, many untreated candidates, no cutoff/instrument | Synthetic control |
| None of the above hold and no randomization is possible | Report correlational findings as such — don't force a causal method |

## Cheat sheet

| Method | Core assumption | Main threat |
|---|---|---|
| DiD | Parallel trends absent treatment | Diverging trends for other reasons |
| IV | Relevance + exclusion restriction | Weak or invalid instrument |
| RD | Continuity of confounders across cutoff | Manipulation of running variable near cutoff |
| Synthetic control | Good pre-period fit implies good counterfactual | Poor donor pool / pre-fit |

## Exercise

Take the DiD example above and add a third pre-period point per group.
Plot both groups' trajectories and assess visually whether parallel
trends looks plausible. Then re-estimate the DiD coefficient using only
two of the three pre-periods and compare — how sensitive is the estimate
to which pre-periods you include?
