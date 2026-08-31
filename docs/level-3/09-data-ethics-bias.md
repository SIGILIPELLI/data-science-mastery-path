# 09 · Data Ethics & Bias

A model can be statistically excellent and still cause real harm — because
the harm usually isn't a bug in the math, it's baked into the data, the
labels, or the choice of what to optimize. This module covers the concrete
checks a data scientist should run before shipping a model that touches
people: where bias enters a pipeline, how to measure it, and what to do
about it.

## Where bias actually enters a pipeline

```text
1. Historical data     → reflects past decisions, which may have been biased
2. Sampling            → who is over/under-represented in your dataset
3. Labels              → proxies for the real target, chosen by humans
4. Features            → correlate with protected attributes even if excluded
5. Objective function  → optimizing accuracy ≠ optimizing fairness
6. Deployment context  → a fair model used unfairly is still a problem
```

The most common mistake is believing step 4 is solved by "we didn't include
race/gender as a feature." Removing a protected attribute doesn't remove
its information from the data if other features (zip code, name, school)
correlate with it — a phenomenon usually called **proxy discrimination**.

## A concrete example: loan approval model

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "applicant_id": range(1, 11),
    "zip_code": ["90210", "90011", "90210", "90011", "90210",
                 "90011", "90210", "90011", "90210", "90011"],
    "group": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    "income": [95000, 42000, 88000, 39000, 91000, 45000, 87000, 41000, 93000, 44000],
    "approved": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
})

# zip_code is a near-perfect proxy for group here
print(pd.crosstab(df["zip_code"], df["group"]))
```

```text
group     A  B
zip_code
90011     0  5
90210     5  0
```

Dropping `group` from the model but keeping `zip_code` reproduces the same
disparate outcome, just indirectly. Checking a candidate feature's
correlation with any known protected/sensitive attribute — even when that
attribute itself won't be modeled — is a standard pre-modeling audit step.

## Measuring group fairness

There is no single "fairness" metric — different definitions can be
mutually incompatible for the same model. Three of the most commonly used:

```python
def group_metrics(df: pd.DataFrame, group_col: str, pred_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        tp = ((sub[pred_col] == 1) & (sub[label_col] == 1)).sum()
        fp = ((sub[pred_col] == 1) & (sub[label_col] == 0)).sum()
        fn = ((sub[pred_col] == 0) & (sub[label_col] == 1)).sum()
        tn = ((sub[pred_col] == 0) & (sub[label_col] == 0)).sum()
        rows.append({
            "group": g,
            "selection_rate": sub[pred_col].mean(),          # demographic parity
            "tpr": tp / (tp + fn) if (tp + fn) else np.nan,  # equal opportunity
            "fpr": fp / (fp + tn) if (fp + tn) else np.nan,  # equalized odds (other half)
        })
    return pd.DataFrame(rows)
```

- **Demographic parity** — equal *selection rate* across groups, regardless
  of actual outcome. Appropriate when the decision itself should be
  group-blind (e.g. ad exposure).
- **Equal opportunity** — equal *true positive rate* across groups: among
  people who should be approved, an equal share are approved regardless of
  group.
- **Equalized odds** — equal TPR *and* equal FPR across groups.

It's a mathematical fact (not a modeling failure) that demographic parity
and equal opportunity generally can't both hold exactly unless base rates
are equal across groups — so the choice of which fairness definition to
target is a policy decision, not a purely technical one, and should be made
explicitly with stakeholders rather than defaulted to by whichever metric
the library reports first.

## Auditing before and after mitigation

```python
from sklearn.linear_model import LogisticRegression

X = df[["income"]]
y = df["approved"]
model = LogisticRegression().fit(X, y)
df["pred"] = model.predict(X)

print(group_metrics(df, "group", "pred", "approved"))
```

```text
  group  selection_rate  tpr  fpr
0     A             1.0  1.0  NaN
1     B             0.0  0.0  NaN
```

A selection-rate gap this stark (1.0 vs 0.0) — even without `group` as a
feature — confirms the proxy effect from `income` (which is correlated with
`group` in this toy data): the audit step is what surfaces the problem, not
avoidance of the feature by name.

## Mitigation options, roughly in order of invasiveness

```text
1. Reweight training examples so groups contribute equally to the loss
2. Remove or transform proxy features found by correlation audit
3. Post-process thresholds per group to equalize a chosen fairness metric
4. Change the label/target itself if it's a poor proxy for the real goal
5. Decide the model shouldn't be deployed for this decision at all
```

Option 5 is a legitimate outcome of a fairness audit, not a failure of the
project — some decisions (e.g. high-stakes ones with thin, biased
historical data) may not currently be automatable responsibly regardless of
technique.

## Cheat sheet

| Concept | What to check |
|---|---|
| Proxy discrimination | Correlate candidate features with protected attributes, even if unused |
| Demographic parity | Equal selection rate across groups |
| Equal opportunity | Equal TPR across groups (for people who should be selected) |
| Equalized odds | Equal TPR and FPR across groups |
| Impossibility results | Parity and opportunity can't both hold exactly if base rates differ |
| Mitigation | Reweighting → feature removal → threshold adjustment → target redefinition → don't deploy |

## Exercise

Take a classifier from an earlier module (or the toy `df` above). Pick a
sensitive attribute you have or can simulate, compute selection rate, TPR,
and FPR per group, and write two sentences: which fairness definition
matters most for this specific decision, and why the other definitions
would be the wrong ones to optimize for here.
