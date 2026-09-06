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

## How It Actually Works

A **two-proportion z-test** treats each group's conversion rate as a sample
proportion `p̂ = conversions / n`, which is itself the mean of a bunch of
0/1 (Bernoulli) outcomes. By the Central Limit Theorem, `p̂` is
approximately normally distributed with standard error
`SE = sqrt(p(1-p)/n)`. Under the null hypothesis that both groups truly have
the same underlying conversion rate `p`, the test pools both groups'
successes to estimate that common `p`, computes the pooled standard error of
the *difference* `p̂₁ - p̂₂`, and reports
`z = (p̂₁ - p̂₂) / SE_pooled`. Because `z` is (approximately) a draw from a
standard normal distribution when the null is true, converting it to a
p-value is a table lookup: `p = 2 * (1 - Φ(|z|))` for a two-sided test,
where `Φ` is the standard normal CDF.

**Sample size calculations** invert this same formula to solve for `n`
given a target detectable effect size, significance level (`α`, usually
0.05 — your false-positive tolerance), and power (`1-β`, usually 0.80 — your
probability of detecting the effect if it's real). Both `α` and `power`
correspond to critical z-values (`z_α/2 ≈ 1.96`, `z_β ≈ 0.84` for the
conventional settings), and the required `n` per group grows roughly as
`(z_α/2 + z_β)² × [p₁(1-p₁) + p₂(1-p₂)] / (p₁-p₂)²` — meaning the required
sample size grows with the *square* of how small an effect you want to
detect, which is why detecting a 0.5 percentage-point lift can need 10-20x
the sample of detecting a 5-point lift.

**The confidence interval** on the lift is built from the same standard
error but without pooling (since it's not testing a null of "no
difference," it's estimating the actual gap): `(p̂₁-p̂₂) ± z_α/2 × SE`. A
95% CI has the specific long-run interpretation that if you reran this exact
experiment many times, about 95% of such intervals would contain the true
population difference — it does *not* mean "95% probability the true value
is in this particular interval," a common misreading.

The **multiple-comparisons** warning is a direct consequence of how p-values
work: testing at `α=0.05` means a 5% false-positive rate *per test* under
the null. Testing `k` independent metrics gives a chance of at least one
false positive of `1 - (1-0.05)^k`, which is already ~40% at `k=10` — not
because anything is wrong with any individual test, but because you're
effectively buying 10 separate lottery tickets for a false "win."

## Exercise

Re-run the simulation above with `n_per_group = 500` instead of 4,000
(same true rates). Report the p-value and confidence interval — do you
still detect the effect? Explain what this demonstrates about the sample
size calculation earlier in the module.
