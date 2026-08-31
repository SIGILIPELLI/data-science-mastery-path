# 02 · Statistical Testing Deep Dive

Level 1 introduced the t-test. Real datasets throw more shapes at you:
categorical vs. categorical, more than two groups, non-normal
distributions. This module builds the decision tree for "which test do I
run?"

## Chi-square test: are two categorical variables related?

```python
import numpy as np
import pandas as pd
from scipy import stats

# Rows = device type, columns = converted (0/1) — observed counts
table = pd.DataFrame(
    [[180, 20], [150, 50]],
    index=["Desktop", "Mobile"],
    columns=["No purchase", "Purchase"],
)
print(table)

chi2, p, dof, expected = stats.chi2_contingency(table)
print(round(chi2, 3), round(p, 5), dof)
```

```text
         No purchase  Purchase
Desktop          180        20
Mobile           150        50
14.815 0.00012 1
```

The chi-square test compares the *observed* counts in a contingency table
against the counts you'd *expect* if the two variables were independent. A
tiny p-value (0.00012) says: purchase rate really does depend on device
type here (desktop converts at 10%, mobile at 25%) — this is not
explainable by chance alone. Use chi-square whenever both variables are
categorical (e.g. "does plan tier relate to churn?").

## One-way ANOVA: comparing more than two group means

A t-test only compares two groups. For three or more, use ANOVA — it tests
whether *at least one* group mean differs from the rest:

```python
rng = np.random.default_rng(1)
control  = rng.normal(50, 8, 40)
variant_a = rng.normal(53, 8, 40)
variant_b = rng.normal(58, 8, 40)

f_stat, p_val = stats.f_oneway(control, variant_a, variant_b)
print(round(f_stat, 3), round(p_val, 5))
```

```text
9.936 9e-05
```

A significant ANOVA (p < 0.05) tells you *some* difference exists among the
three groups, but not *which* pair differs. For that you need a post-hoc
test with a multiple-comparisons correction (e.g. Tukey's HSD), because
running three separate t-tests without correction inflates your false
positive rate — exactly the p-hacking trap from Level 1.

```python
from itertools import combinations

groups = {"control": control, "a": variant_a, "b": variant_b}
alpha_corrected = 0.05 / 3  # Bonferroni: 3 pairwise comparisons

for (name1, g1), (name2, g2) in combinations(groups.items(), 2):
    t, p = stats.ttest_ind(g1, g2)
    flag = "significant" if p < alpha_corrected else "not significant"
    print(f"{name1} vs {name2}: p={p:.4f} -> {flag}")
```

```text
control vs a: p=0.0433 -> not significant
control vs b: p=0.0000 -> significant
b vs a: p=0.0113 -> not significant
```

With the Bonferroni-corrected threshold (0.0167 instead of 0.05),
`control vs a` no longer clears the bar even though its raw p-value looked
promising — this is the correction doing its job.

## Non-parametric alternatives: when data isn't normal

The t-test and ANOVA assume roughly normal data. When a distribution is
heavily skewed (e.g. revenue, wait times) or you have a small sample,
**Mann-Whitney U** is the non-parametric analogue of the two-sample t-test —
it compares *ranks* instead of means:

```python
skewed_a = rng.exponential(scale=5, size=30)
skewed_b = rng.exponential(scale=7, size=30)

u_stat, p_val = stats.mannwhitneyu(skewed_a, skewed_b, alternative="two-sided")
print(round(u_stat, 1), round(p_val, 4))
print(round(np.median(skewed_a), 2), round(np.median(skewed_b), 2))
```

```text
337.0 0.1682
2.98 4.7
```

Even though the medians look different (2.98 vs 4.7), the Mann-Whitney test
says this could plausibly be chance with only 30 samples per group and this
much variance in an exponential distribution.

## Choosing a test: decision guide

| Situation | Test |
|---|---|
| 2 groups, continuous outcome, roughly normal | `stats.ttest_ind` |
| 2 groups, continuous outcome, skewed / small n | `stats.mannwhitneyu` |
| 3+ groups, continuous outcome | `stats.f_oneway` (ANOVA), then post-hoc |
| 2 categorical variables | `stats.chi2_contingency` |
| Paired before/after measurements | `stats.ttest_rel` |

## Exercise

Generate three groups of 25 samples each from `rng.normal` with means 10,
12, and 11 (same std of 3). Run a one-way ANOVA. If it's significant, run
all three pairwise t-tests with a Bonferroni correction and report which
pairs (if any) survive correction. Then repeat the whole experiment with
group means 10, 10.5, and 10.2 — what happens to the ANOVA result, and why
does that make sense?
