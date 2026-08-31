# 05 · Advanced Experimentation at Scale

A single A/B test with one metric and one comparison is straightforward.
An org running hundreds of concurrent experiments across shared user
populations, with dozens of tracked metrics each, runs into an entirely
different set of problems: statistical ones (multiple comparisons,
interference) and operational ones (overlapping experiments, novelty
effects). This module covers both.

## The multiple comparisons problem

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)

# Simulate 20 metrics with NO real effect (all null)
false_positives = 0
for _ in range(20):
    control = rng.normal(100, 15, 1000)
    treatment = rng.normal(100, 15, 1000)  # same distribution — no real effect
    _, p = stats.ttest_ind(control, treatment)
    if p < 0.05:
        false_positives += 1

print(false_positives)  # expect ~1 out of 20 by chance alone, at alpha=0.05
```

```text
1
```

At a 5% significance threshold, testing 20 unrelated metrics on data with
no real effect anywhere yields roughly one "significant" result by chance
alone. A team that runs one experiment but checks 20 metrics, then reports
the one that came back significant, has effectively run 20 tests and
reported the false positive — this is why a single experiment needs one
pre-registered primary metric, with the rest labeled explicitly as
secondary/exploratory.

## Correcting for multiple comparisons

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.04, 0.03, 0.20, 0.049, 0.11]

reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
print(reject)
print(p_corrected.round(3))
```

```text
[ True False False False False False]
[0.006 0.12  0.12  0.24  0.147 0.165]
```

Benjamini-Hochberg (FDR control, `fdr_bh`) is generally preferred over
Bonferroni for exploratory metric panels — Bonferroni controls the
probability of *any* false positive and gets extremely conservative as
the number of metrics grows; FDR control instead bounds the *expected
proportion* of false positives among findings called significant, which
is usually the more useful guarantee when scanning a metrics dashboard
for effects.

## Interference between experiments

```text
Problem: Experiment A tests a new recommendation algorithm.
         Experiment B tests a new checkout flow.
         Same users see BOTH — if A changes what's in the cart,
         B's checkout-flow effect is contaminated by A's effect.

Mitigations:
1. Mutually exclusive experiment layers for interacting changes
   (users in experiment A's layer are excluded from B's layer)
2. Orthogonal layering when changes are unlikely to interact
   (a UI color test and a backend caching test can usually run concurrently)
3. Interaction effect tests — deliberately run a 2x2 to check
   if the combination behaves differently than either alone
```

```python
# 2x2 factorial design to explicitly test for interaction
import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    "algo_variant": ["old", "old", "new", "new"] * 250,
    "checkout_variant": (["old", "new"] * 500),
    "conversion": rng.binomial(1, 0.12, 1000),
})

model = smf.logit(
    "conversion ~ C(algo_variant) * C(checkout_variant)", data=df
).fit(disp=0)
print(model.pvalues)
```

The interaction term (`algo_variant:checkout_variant`) tells you directly
whether the two changes combine super-additively, sub-additively, or
independently — running them in separate silos without ever checking this
term means shipping both and being surprised when the combined effect
doesn't match the sum of two isolated experiment reads.

## Novelty and primacy effects

```text
Novelty effect: a new feature gets a temporary engagement boost simply
                because it's new/different — decays over days-to-weeks
Primacy effect: a UI change is temporarily under-used because users are
                used to the old layout — recovers as users adapt

Mitigation: look at the metric trend OVER TIME within the experiment,
            not just the endpoint average. A treatment effect that's
            large on day 1 and decaying toward zero by day 14 is a
            novelty effect, not a durable lift.
```

```python
weekly_lift = pd.Series([0.08, 0.05, 0.02, 0.01, 0.01], index=[f"week_{i}" for i in range(1, 6)])
print(weekly_lift)
```

```text
week_1    0.08
week_2    0.05
week_3    0.02
week_4    0.01
week_5    0.01
dtype: float64
```

A decaying-but-nonzero lift like this one is often reported honestly as
"a durable ~1% lift, with a temporary novelty spike in week 1" — a team
that only ran the experiment for a week and reported the 8% figure would
be materially overstating the long-run impact.

## Sequential testing and peeking

```text
Problem: checking a test's p-value daily and stopping as soon as it
         crosses 0.05 inflates the false positive rate far above 5% —
         you're implicitly running many tests (one per day checked)
         and taking the best one.

Fix: either commit to a fixed sample size / duration set BEFORE the test
     (classical approach), or use a sequential testing method (e.g.
     mSPRT, alpha spending) explicitly designed to allow valid continuous
     monitoring without inflating false positive rate.
```

## Cheat sheet

| Problem | Cause | Fix |
|---|---|---|
| False "significant" metrics | Checking many metrics, one pre-registered primary | FDR correction (`fdr_bh`) on secondary metrics |
| Contaminated experiment reads | Overlapping experiments on shared users | Mutually exclusive layers, or explicit interaction test |
| Overstated long-run lift | Novelty/primacy effect | Look at metric trend over time, not just endpoint average |
| Inflated false positive rate | Peeking at p-values daily and stopping early | Fixed duration, or a proper sequential testing method |

## Exercise

Simulate two experiments sharing a user population (as in the 2x2 example
above) where one variant genuinely interacts with the other (bake in a
real interaction effect this time, not `rng.binomial` for both cells
independently). Fit the interaction model and confirm the interaction
term is statistically detectable — then write two sentences on what
running these as fully separate, non-overlapping experiments would have
hidden.
