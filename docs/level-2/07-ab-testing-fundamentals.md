# 07 · A/B Testing Fundamentals

Level 1 covered hypothesis testing in the abstract. A/B testing is that
theory applied to the most common business question: "did this change
actually help?" This module covers the design and analysis of a simple
two-group experiment.

## Designing the test: what are you actually comparing?

An A/B test needs a clearly defined **unit of randomization** (usually
users, not page views — the same user shouldn't see both variants), a
**primary metric** decided *before* the test starts, and a **minimum
detectable effect (MDE)** — the smallest lift you actually care about
finding.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_per_group = 4000

# Control: 8% conversion. Treatment: true 9.5% conversion (a real 1.5pp lift).
control = np.random.binomial(1, 0.08, n_per_group)
treatment = np.random.binomial(1, 0.095, n_per_group)

df = pd.DataFrame({
    "group": ["control"] * n_per_group + ["treatment"] * n_per_group,
    "converted": np.concatenate([control, treatment]),
})
print(df.groupby("group")["converted"].agg(["mean", "count"]))
```

```text
             mean  count
group
control    0.0785   4000
treatment  0.0975   4000
```

## Sample size: how many users do you need?

Before running a test, estimate the sample size needed to detect your MDE
with reasonable power (typically 80%) at a chosen significance level
(typically 5%).

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

effect = proportion_effectsize(0.08, 0.095)   # Cohen's h
analysis = NormalIndPower()
n_required = analysis.solve_power(effect_size=effect, alpha=0.05, power=0.8, ratio=1)
print(round(n_required))
```

```text
3775
```

Detecting a 1.5 percentage-point lift on an 8% baseline needs about 3,775
users per group — this experiment (4,000 per group) is adequately powered.
Skipping this step is the single most common A/B testing mistake: an
underpowered test that finds "no significant difference" often just didn't
have enough users to detect a real effect.

## Running the test: a two-proportion z-test

```python
from statsmodels.stats.proportion import proportions_ztest

successes = df.groupby("group")["converted"].sum()
counts = df.groupby("group")["converted"].count()

z_stat, p_value = proportions_ztest(
    [successes["treatment"], successes["control"]],
    [counts["treatment"], counts["control"]],
)
print(f"z = {z_stat:.2f}, p = {p_value:.4f}")
```

```text
z = 2.26, p = 0.0239
```

With `p = 0.024 < 0.05`, we reject the null hypothesis of no difference —
the observed lift is unlikely to be due to chance alone.

## Confidence interval on the lift

A p-value tells you *whether* there's an effect; a confidence interval
tells you *how big* it plausibly is — usually the more actionable number
for a launch decision.

```python
from statsmodels.stats.proportion import confint_proportions_2indep

ci_low, ci_high = confint_proportions_2indep(
    successes["treatment"], counts["treatment"],
    successes["control"], counts["control"],
)
print(f"Lift 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
```

```text
Lift 95% CI: [0.0026, 0.0354]
```

The interval excludes zero (consistent with the significant p-value) and
tells stakeholders the plausible range: the true lift is likely between
0.26 and 3.54 percentage points — useful for a revenue-impact estimate,
which a bare p-value can't give you.

## Common pitfalls

- **Peeking**: checking significance daily and stopping as soon as
  `p < 0.05` inflates the false-positive rate dramatically. Decide the
  sample size (or test duration) in advance and stick to it, or use a
  sequential-testing method designed for early stopping.
- **Multiple metrics**: testing 10 metrics at `α = 0.05` gives roughly a
  40% chance one is "significant" by chance alone. Pre-register a single
  primary metric; treat others as secondary/exploratory.
- **Novelty effects**: a lift in week one can fade by week three as users
  adjust to the change — run long enough to see if the effect holds.

## Cheat sheet

| Task | Code |
|---|---|
| Sample size for a proportion test | `NormalIndPower().solve_power(effect_size, alpha, power)` |
| Two-proportion z-test | `proportions_ztest([s1, s2], [n1, n2])` |
| CI on the difference | `confint_proportions_2indep(...)` |
| Effect size (Cohen's h) | `proportion_effectsize(p1, p2)` |

## Exercise

Re-run the simulation above with `n_per_group = 500` instead of 4,000
(same true rates). Report the p-value and confidence interval — do you
still detect the effect? Explain what this demonstrates about the sample
size calculation earlier in the module.
