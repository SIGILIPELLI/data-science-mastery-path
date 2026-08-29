# 06 · Statistics Fundamentals

Every "is this real, or is it noise?" question in data science is answered
with statistics. This module covers the vocabulary you'll use constantly:
mean/variance, distributions, hypothesis testing, confidence intervals, and
the single most dangerous trap in applied statistics — p-hacking.

## Mean, variance, and standard deviation

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(5)
sample = rng.normal(50, 10, 30)

print(sample.mean().round(2))          # 47.83
print(sample.std(ddof=1).round(2))     # 8.82
print(np.median(sample).round(2))      # 47.47
```

```text
47.83
8.82
47.47
```

`ddof=1` (delta degrees of freedom) tells NumPy to divide by `n-1` instead of
`n` — the standard correction for estimating a population's spread from a
*sample* rather than measuring the whole population. Always use `ddof=1` for
sample data, which is almost all data you'll ever have.

Note the sample mean (47.83) and median (47.47) differ slightly from the
true generating mean (50) — that's expected sampling variability with only
30 points, not a bug. This is the core intuition behind everything else in
this module: **a sample statistic is an estimate, not the truth**, and
smaller samples wobble more.

## The normal distribution and why it's everywhere

Many real-world quantities — heights, measurement errors, sums of many
independent small effects — approximate a **normal (Gaussian) distribution**:
symmetric, bell-shaped, fully described by just its mean and standard
deviation. The practical rule of thumb (the "68-95-99.7 rule"):

| Range | % of data (approx.) |
|---|---|
| mean ± 1 std | 68% |
| mean ± 2 std | 95% |
| mean ± 3 std | 99.7% |

This is *why* the IQR outlier rule from Module 04 works, and why a value
more than ~2-3 standard deviations from the mean is worth a second look.

## Hypothesis testing: is a difference real?

Suppose you're comparing two groups (e.g. two marketing campaigns) and see
different average outcomes. A **t-test** asks: how likely is a difference
this large (or larger) if there were actually no true difference at all?

```python
group_a = rng.normal(50, 10, 40)
group_b = rng.normal(55, 10, 40)

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(t_stat.round(3), p_val.round(4))
print(group_a.mean().round(2), group_b.mean().round(2))
```

```text
-1.656 0.1017
49.0 52.41
```

The **p-value** (0.1017 here) is the probability of seeing a difference this
extreme by pure chance *if the two groups were truly identical*. The
conventional cutoff is **p < 0.05** ("statistically significant"). Here,
even though group B's sample mean (52.41) is visibly higher than group A's
(49.0), p = 0.10 means this difference is *not* strong enough evidence to
rule out chance — with only 40 samples per group, a gap this size can easily
happen even when the underlying populations don't truly differ.

!!! warning "A p-value is not the probability your hypothesis is true"
    p = 0.10 does **not** mean "there's a 90% chance the difference is
    real." It means "if there were truly no difference, data this extreme
    would appear about 10% of the time by chance alone." This distinction
    trips up even experienced analysts — always phrase results in terms of
    what the test actually measures.

## The p-hacking trap

Here's the single most important statistical trap for a working data
scientist to internalize: **if you run enough tests on pure noise, some will
look "significant" purely by chance.**

```python
sig_count = 0
trials = 100
for i in range(trials):
    x = rng.normal(0, 1, 20)     # pure random noise
    y = rng.normal(0, 1, 20)     # pure random noise, independent of x
    _, p = stats.ttest_ind(x, y)
    if p < 0.05:
        sig_count += 1

print(sig_count, trials)
```

```text
9 100
```

Out of 100 comparisons between two groups that are **both pure random
noise with zero true difference**, 9 came back "statistically significant"
at the p < 0.05 threshold — close to the ~5% false-positive rate the
threshold is designed to produce. This is **p-hacking**: if you slice a
dataset 20 different ways ("what about just mobile users?" "what about just
last week?") and report only the slice that hit p < 0.05, you are
guaranteed to eventually find a "significant" result that's actually noise.
The fix: decide your hypothesis and test *before* looking at the data, and
if you must run many comparisons, correct for it (e.g. Bonferroni
correction — divide your significance threshold by the number of tests).

## Confidence intervals

A **confidence interval (CI)** gives a range of plausible values for a true
population parameter, not just a single point estimate:

```python
mean = sample.mean()
sem = stats.sem(sample)
ci = stats.t.interval(0.95, len(sample) - 1, loc=mean, scale=sem)
print(ci)
```

```text
(44.54, 51.13)
```

Read this as: "we're 95% confident the true population mean falls between
44.54 and 51.13." Note this range does contain the true generating mean
(50) — as it should, most of the time, for a correctly-computed 95% CI. A
narrower interval (from a larger sample) is a more precise estimate; a wider
one (small sample) is more honest about how little you actually know.

## Cheat sheet

| Concept | Code | What it tells you |
|---|---|---|
| Sample mean/std | `x.mean()`, `x.std(ddof=1)` | Center and spread of your data |
| t-test | `stats.ttest_ind(a, b)` | Is a difference between two groups likely real? |
| p-value | second output of `ttest_ind` | P(data this extreme \| no true difference) |
| Confidence interval | `stats.t.interval(0.95, df, loc, scale)` | Plausible range for the true value |
| p-hacking | — | Testing many slices and reporting only the "hits" |
| Bonferroni fix | `alpha / n_tests` | Adjusted threshold for multiple comparisons |

## Exercise

Re-run the p-hacking simulation with `trials = 1000` and `alpha = 0.01`
(count `p < 0.01` instead of `p < 0.05`). What fraction of the 1000
noise-vs-noise comparisons come back "significant"? Compare that fraction to
the threshold itself, and explain in one sentence why they should be close.
