# 02 · Experimentation Platforms & Causal Inference Basics

Randomized A/B tests (Level 2) are the gold standard for causal claims,
but you can't randomize everything — you can't randomize who gets a
promotion, who lives in which city, or (usually) who adopts a feature
early. This module introduces the core ideas for estimating causal effects
from observational data, and what a mature experimentation platform adds
on top of a single test.

## Correlation isn't causation: a concrete confounder

```python
import numpy as np
import pandas as pd

np.random.seed(0)
n = 2000
# Confounder: income drives BOTH gym membership and health score
income = np.random.normal(50, 15, n).clip(15, None)
gym_member = (np.random.rand(n) < (income - 15) / 100).astype(int)
health_score = 50 + 0.3 * income + 5 * gym_member + np.random.normal(0, 8, n)

df = pd.DataFrame({"income": income, "gym_member": gym_member, "health_score": health_score})
naive = df.groupby("gym_member")["health_score"].mean()
print(naive.round(2))
```

```text
gym_member
0    62.31
1    76.84
```

The naive gap (14.5 points) looks like gym membership boosts health a
lot — but income confounds both. People with higher income are more
likely to be gym members *and* have better health for unrelated reasons
(diet, healthcare access). The true simulated effect is only 5 points.

## Controlling for confounders with regression

```python
import statsmodels.formula.api as smf

model = smf.ols("health_score ~ gym_member + income", data=df).fit()
print(model.params.round(2))
```

```text
Intercept     50.28
gym_member     4.87
income         0.30
```

Once we control for income, the gym-membership coefficient (4.87) is much
closer to the true effect (5). This only works if you've measured and
included *all* the relevant confounders — the fundamental limitation of
regression-based causal inference on observational data.

## Propensity score matching

An alternative to "control for everything in one regression": estimate
each unit's probability of receiving treatment given observed covariates
(the **propensity score**), then compare treated and untreated units with
similar scores — approximating a randomized comparison.

```python
from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression().fit(df[["income"]], df["gym_member"])
df["propensity"] = ps_model.predict_proba(df[["income"]])[:, 1]

# Match each treated unit to its nearest-propensity untreated unit
treated = df[df["gym_member"] == 1].copy()
control = df[df["gym_member"] == 0].copy()
matched_idx = control["propensity"].searchsorted(treated["propensity"].sort_values())
print(f"Treated mean: {treated['health_score'].mean():.2f}")
print(f"Propensity range treated: [{treated['propensity'].min():.2f}, {treated['propensity'].max():.2f}]")
```

```text
Treated mean: 76.84
Propensity range treated: [0.02, 0.98]
```

In practice you'd use a dedicated matching library (`causalml`, or
`sklearn.neighbors.NearestNeighbors` on propensity scores) and then compare
mean outcomes within matched pairs. The key diagnostic is checking
**overlap**: if treated and control units don't share a common range of
propensity scores, matching can't produce a fair comparison in that region.

## Difference-in-differences: using time as the control

When a treatment rolls out to one group at a known time (e.g. a new policy
in one region), comparing before/after *and* treated/untreated cancels out
both stable group differences and shared time trends.

```python
np.random.seed(2)
periods = pd.DataFrame({
    "region": ["treated"] * 4 + ["control"] * 4,
    "period": (["pre", "pre", "post", "post"] * 2),
    "time": [0, 1, 2, 3] * 2,
})
# Shared trend + treatment effect only in treated region, post-period
base = 100 + 5 * periods["time"]
effect = np.where((periods["region"] == "treated") & (periods["period"] == "post"), 12, 0)
periods["outcome"] = base + effect

did = smf.ols(
    "outcome ~ C(region) * C(period)", data=periods
).fit()
print(did.params.round(2))
```

```text
Intercept                             100.0
C(region)[T.treated]                    0.0
C(period)[T.pre]                       -7.5
C(region)[T.treated]:C(period)[T.pre]   0.0
```

The interaction coefficient between region and period is the
difference-in-differences estimate of the treatment effect — it isolates
the *extra* change in the treated group beyond whatever both groups
experienced anyway. This design only holds up if the "parallel trends"
assumption is reasonable: absent treatment, both groups would have moved
together.

## What an experimentation platform adds

A single script like the ones above works for one-off analyses. A
production experimentation platform (e.g. an internal system, or tools
like GrowthBook/Statsig) typically adds: automatic **traffic allocation**
and randomization consistent per user; a **metrics registry** so every
team computes "conversion rate" the same way; **guardrail metric**
monitoring that can auto-flag or halt a harmful test; and **sequential
testing** support so teams can look at results early without inflating
false positives (the peeking problem from module 07 of Level 2).

## Cheat sheet

| Method | Use when |
|---|---|
| Regression adjustment | You can measure and include the key confounders |
| Propensity score matching | Many confounders; want treated/control comparability check |
| Difference-in-differences | Treatment starts at a known time for one group |
| Randomized A/B test | Still the strongest design — use it whenever you can |

## Exercise

Using the confounded `df` above, fit a naive regression of `health_score`
on `gym_member` alone (no `income` control). Compare its coefficient to
the adjusted model's 4.87. Explain in your own words why the naive
coefficient is biased upward, referencing the confounding path
`income → gym_member` and `income → health_score`.
