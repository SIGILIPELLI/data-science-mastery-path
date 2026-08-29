# 10 · Project — End-to-End Data Analysis Report

This capstone runs the full workflow from Module 01 on one dataset: a
synthetic subscription product's signup/engagement/churn data. The goal —
"why are customers churning, and what should the team do about it?" — mirrors
a real first-week task at most data science jobs.

## Step 1 — get and inspect the data

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

rng = np.random.default_rng(21)
n = 300

signup_channel = rng.choice(["organic", "paid_ad", "referral"], n, p=[0.5, 0.3, 0.2])
tenure_days = rng.integers(1, 365, n)
base_engagement = rng.normal(20, 6, n)
channel_boost = np.where(signup_channel == "referral", 6,
                 np.where(signup_channel == "paid_ad", -2, 0))
engagement_score = base_engagement + channel_boost + tenure_days * 0.01
plan = rng.choice(["free", "pro"], n, p=[0.7, 0.3])
churn_prob = np.clip(0.35 - 0.01 * engagement_score + np.where(plan == "pro", -0.05, 0), 0.02, 0.9)
churned = rng.binomial(1, churn_prob)

df = pd.DataFrame({
    "signup_channel": signup_channel,
    "tenure_days": tenure_days,
    "engagement_score": engagement_score.round(2),
    "plan": plan,
    "churned": churned,
})
print(df.shape)
print(df.isna().sum().sum())
print(df.describe())
```

```text
(300, 5)
0
       tenure_days  engagement_score     churned
count   300.000000        300.000000  300.000000
mean    175.840000         22.708500    0.133333
std     106.853451          6.531107    0.340503
min       1.000000          5.470000    0.000000
25%      83.750000         18.350000    0.000000
50%     177.500000         22.440000    0.000000
75%     267.500000         26.977500    0.000000
max     362.000000         39.710000    1.000000
```

No missing data (Module 03's `isna().sum()` check), 300 rows, 5 columns.
Overall churn rate is 13.3% (`churned` mean). Nothing in `.describe()` looks
like an obvious outlier or broken value, so we can move to EDA.

## Step 2 — EDA: does churn differ by group?

```python
print(df.groupby("signup_channel")["churned"].mean().round(3))
print(df.groupby("plan")["churned"].mean().round(3))
```

```text
signup_channel
organic     0.141
paid_ad     0.146
referral    0.091
Name: churned, dtype: float64
plan
free    0.156
pro     0.089
Name: churned, dtype: float64
```

Two candidate findings jump out: **referral signups churn less** (9.1% vs.
~14% for organic/paid), and **pro-plan users churn less** than free users
(8.9% vs. 15.6%). Both are directional — Module 06's tools are needed to
know whether they're real or noise.

## Step 3 — is the engagement/churn relationship real?

```python
churned_eng = df[df["churned"] == 1]["engagement_score"]
active_eng = df[df["churned"] == 0]["engagement_score"]

t, p = stats.ttest_ind(churned_eng, active_eng)
print(t.round(3), p.round(5), churned_eng.mean().round(2), active_eng.mean().round(2))
```

```text
-2.48 0.0137 20.34 23.07
```

Churned customers average an engagement score of 20.34, versus 23.07 for
active customers — and `p = 0.0137`, well under the conventional 0.05
threshold. This is genuine evidence (not proof of causation — see the
warning below) that lower engagement is associated with higher churn.

## Step 4 — quantify the relationship with regression

```python
X = sm.add_constant(df["engagement_score"])
y = df["churned"]
model = sm.OLS(y, X).fit()

print(model.params.round(4))
print(model.rsquared.round(3))
print(model.pvalues.round(5))
```

```text
const               0.3017
engagement_score   -0.0074

R² = 0.02
const               0.00003
engagement_score    0.01370
```

The fitted line: `P(churn) ≈ 0.302 − 0.0074 × engagement_score` — each extra
point of engagement is associated with about a 0.74 percentage-point drop in
churn probability, and the slope's p-value (0.0137) confirms this matches
the t-test above. But **R² = 0.02**: engagement score alone explains only 2%
of the variance in churn. This is an honest, important negative finding —
engagement matters statistically, but it is nowhere near a complete
explanation of who churns, and a model built on this feature alone would
predict poorly.

!!! warning "Correlation, not causation, and check for confounders"
    Per Module 07, this analysis cannot claim "raising engagement will
    reduce churn" — it only shows they're associated. `plan` (pro vs. free)
    is a plausible confounder here worth checking with a
    `groupby(["plan", <engagement bucket>])["churned"].mean()` breakdown,
    the same technique used to catch Simpson's paradox, before recommending
    any engagement-boosting intervention.

## Step 5 — visualize the headline finding

```python
fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("signup_channel")["churned"].mean().plot(kind="bar", ax=ax)
ax.set_ylim(0, 1)          # zero-based axis, per Module 09
ax.set_ylabel("Churn rate")
fig.savefig("churn_by_channel.png", dpi=150)
```

The zero-based y-axis (0 to 1) is deliberate, per Module 09: it correctly
shows that while referral (9.1%) beats organic/paid (~14%), all three
channels' churn rates are much closer together than a truncated axis would
visually suggest.

## Step 6 — the report

A finished write-up, following Module 09's structure:

> **Finding:** Referral signups churn at roughly two-thirds the rate of
> organic or paid signups (9.1% vs. ~14%), and pro-plan users churn at
> roughly half the rate of free users (8.9% vs. 15.6%). Engagement score is
> statistically associated with churn (p = 0.014) but explains only ~2% of
> its variance on its own — not enough to build a churn-prediction model on
> alone.
>
> **Recommendation:** Investing further in the referral channel and in
> free-to-pro conversion look like the two highest-leverage levers on churn
> in this dataset. Engagement is a real but weak signal — worth including
> alongside other features in a future model (Level 2's "Intro to Machine
> Learning for Data Science" module), not as a standalone lever.
>
> **Caveat:** This is observational data — none of these relationships have
> been tested with a controlled experiment, so we cannot yet claim that
> *changing* engagement, plan, or channel *causes* a change in churn. The
> A/B Testing module in Level 2 covers how to test that properly.

## What this project ties together

| Skill | Module | Used here for |
|---|---|---|
| Cleaning/checking | 03 | `isna()`, dtype checks before analysis |
| EDA | 04 | `describe()`, groupby comparisons |
| Visualization | 05, 09 | Zero-based bar chart of churn by channel |
| Hypothesis testing | 06 | t-test on engagement vs. churn |
| Regression | 07 | Quantifying the engagement/churn relationship, R² |
| Communicating findings | 09 | The three-part report structure above |

## Exercise

Extend this analysis with one more variable: `tenure_days`. Run the same
t-test (churned vs. active tenure), fit a regression of churn on tenure, and
write one additional sentence for the report above stating whether tenure
is a stronger or weaker churn signal than engagement score, using the actual
R² and p-value you compute.
