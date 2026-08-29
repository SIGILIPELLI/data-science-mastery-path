# 07 · Correlation & Basic Regression

Correlation quantifies how strongly two numeric variables move together;
regression fits a line through the relationship so you can predict one from
the other. Both are essential tools — and both are widely misused. This
module covers the mechanics, plus the two traps that catch people most
often: correlation-does-not-imply-causation, and Simpson's paradox.

## Pearson correlation

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(9)
n = 100

# Two variables that both trend upward over time, unrelated to each other
ice_cream_sales = rng.normal(200, 40, n) + np.arange(n) * 2
drowning_incidents = rng.normal(5, 2, n) + np.arange(n) * 0.05

r, p = stats.pearsonr(ice_cream_sales, drowning_incidents)
print(r.round(3), p.round(5))
```

```text
0.441 0.0
```

`r = 0.441` is a moderate positive correlation, and `p ≈ 0` says it's very
unlikely to be chance. **But ice cream doesn't cause drowning.** Both
variables were built here to trend upward with `np.arange(n)` — a stand-in
for a shared underlying driver like the calendar (both rise in summer,
independent of each other). This is the classic textbook illustration of:

!!! danger "Correlation does not imply causation"
    A correlation between X and Y can come from (1) X causes Y, (2) Y causes
    X, (3) a third variable causes both (a **confounder** — here, "it's
    summer"), or (4) pure coincidence in a small sample. Seeing `r` and `p`
    tells you a relationship exists; it tells you *nothing* about which of
    these four explains it. Establishing causation needs either a controlled
    experiment (Level 2's A/B Testing module) or the causal-inference
    techniques in Level 3.

## Simple linear regression

Regression answers a related but different question: "given X, what's my
best prediction of Y, and how good is that prediction?"

```python
import statsmodels.api as sm

sqft = rng.normal(1800, 400, n)
price = 50000 + sqft * 120 + rng.normal(0, 20000, n)

X = sm.add_constant(sqft)          # adds the intercept term
model = sm.OLS(price, X).fit()

print(model.params.round(2))       # [intercept, slope]
print(model.rsquared.round(3))
print(model.pvalues.round(5))
```

```text
[47549.33   121.76]
0.854
[0. 0.]
```

Reading this output:

- **Intercept (47,549)** — the model's predicted price when `sqft = 0`
  (not meaningful on its own here, but required by the equation).
- **Slope (121.76)** — each additional square foot is associated with
  about $121.76 more in price. This recovers the true generating slope
  (120) closely, as expected with 100 clean data points.
- **R² (0.854)** — 85.4% of the variance in price is explained by square
  footage alone. R² ranges 0 (no explanatory power) to 1 (perfect fit).
- **p-values (both ≈ 0)** — both the intercept and slope are statistically
  significant, i.e. very unlikely to be zero by chance.

The fitted line: `price ≈ 47,549 + 121.76 × sqft`. Use `model.predict()`
with new square footage values to get price estimates — but always alongside
the R² and a sense of how much scatter remains around that line.

## Simpson's paradox

This is the trap that makes "just look at the aggregate" actively dangerous:
a trend can reverse completely once you split the data into meaningful
subgroups.

```python
import pandas as pd

dept_a = pd.DataFrame({
    "dept": "A",
    "gender": ["M"] * 100 + ["F"] * 20,
    "admitted": [1] * 80 + [0] * 20 + [1] * 17 + [0] * 3,
})
dept_b = pd.DataFrame({
    "dept": "B",
    "gender": ["M"] * 20 + ["F"] * 100,
    "admitted": [1] * 4 + [0] * 16 + [1] * 22 + [0] * 78,
})
combined = pd.concat([dept_a, dept_b], ignore_index=True)

print(combined.groupby("gender")["admitted"].mean().round(3))
print(combined.groupby(["dept", "gender"])["admitted"].mean().round(3))
```

```text
gender
F    0.325
M    0.700

dept  gender
A     F         0.85
      M         0.80
B     F         0.22
      M         0.20
```

Look closely: **within every single department, women are admitted at a
higher rate than men** (85% vs. 80% in dept A; 22% vs. 20% in dept B). But
the **aggregate** numbers show the opposite — 32.5% for women vs. 70% for
men overall. Nothing in the data is wrong; the reversal happens because
women in this dataset applied disproportionately to department B, which
admits at a much lower rate overall (department choice is the confounder).
Reporting only the aggregate number here would produce a conclusion that is
the exact opposite of the truth in every subgroup.

!!! warning "The fix: always check whether a key breakdown changes the story"
    Whenever a variable (department, region, time period, customer segment)
    plausibly affects both your grouping variable and your outcome, check
    the relationship *within* that variable's categories, not just in
    aggregate. If splitting the data ever reverses your conclusion, report
    the split, not the aggregate.

## Cheat sheet

| Task | Code |
|---|---|
| Correlation + significance | `stats.pearsonr(x, y)` |
| Fit a regression line | `sm.OLS(y, sm.add_constant(x)).fit()` |
| Slope/intercept | `model.params` |
| Variance explained | `model.rsquared` |
| Are coefficients significant? | `model.pvalues` |
| Check for Simpson's paradox | `df.groupby([confounder, group])[outcome].mean()` vs. aggregate |

## Exercise

Using the `sqft`/`price` regression above, add a new confounding variable
`neighborhood` (two categories, e.g. `"urban"` and `"suburban"`) such that
urban homes are both smaller *and* pricier per square foot. Show that the
raw `sqft` vs. `price` correlation looks different (weaker or even the wrong
sign) than the within-neighborhood relationship — you've just reproduced
Simpson's paradox with continuous variables instead of counts.
