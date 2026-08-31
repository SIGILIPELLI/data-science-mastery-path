# 05 · Model Interpretability

A model that predicts well but can't explain *why* is a liability in most
real settings — regulators, stakeholders, and your own debugging process
all need to know what's driving predictions. This module covers the main
tools: coefficients, feature importance, permutation importance, and SHAP.

## Baseline: linear model coefficients

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
n = 1000
loans = pd.DataFrame({
    "income_k": np.random.normal(60, 20, n).clip(15, None),
    "debt_ratio": np.random.uniform(0, 0.6, n),
    "credit_score": np.random.normal(680, 60, n),
})
logit = -6 + 0.03 * loans["income_k"] - 8 * loans["debt_ratio"] + 0.01 * loans["credit_score"]
loans["approved"] = (np.random.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)

X = loans[["income_k", "debt_ratio", "credit_score"]]
X_scaled = StandardScaler().fit_transform(X)
model = LogisticRegression().fit(X_scaled, loans["approved"])
print(pd.Series(model.coef_[0], index=X.columns).round(3))
```

```text
income_k        0.847
debt_ratio     -1.203
credit_score    0.612
```

Because features were standardized first, coefficients are directly
comparable: `debt_ratio` has the largest magnitude effect on the
log-odds of approval, followed by income, then credit score. This kind
of interpretability — one number per feature, sign and magnitude both
meaningful — is the main reason to still reach for linear/logistic models
even when a tree ensemble would predict slightly better.

## Tree-based feature importance (and its blind spot)

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, loans["approved"])
print(pd.Series(forest.feature_importances_, index=X.columns).round(3))
```

```text
income_k        0.31
debt_ratio      0.46
credit_score    0.23
```

Built-in importance (mean decrease in impurity) is fast but biased toward
high-cardinality, continuous features, and it says nothing about
*direction* — you know `debt_ratio` matters, not whether higher debt ratio
raises or lowers approval odds. It's a good first look, not a final
explanation.

## Permutation importance: a more honest alternative

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(forest, X, loans["approved"], n_repeats=20, random_state=0)
print(pd.Series(perm.importances_mean, index=X.columns).round(3))
```

```text
income_k        0.052
debt_ratio      0.089
credit_score    0.031
```

Permutation importance shuffles one feature's values at a time and
measures how much model performance *drops* — a feature the model
genuinely relies on will hurt performance when scrambled. Unlike impurity
importance, this works for any model type (not just trees) and isn't
biased by feature cardinality, at the cost of being slower to compute.

## SHAP values: per-prediction explanations

Global importance answers "what matters overall?" SHAP answers "why did
*this specific* prediction come out this way?" — critical when a customer
asks "why was I denied?"

```python
import shap

explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X.iloc[:5])

# shap_values[1] = contributions toward the "approved" class
for i in range(3):
    print(f"Applicant {i}: base={explainer.expected_value[1]:.3f}, "
          f"contributions={np.round(shap_values[1][i], 3)}")
```

```text
Applicant 0: base=0.520, contributions=[ 0.041 -0.183  0.062]
Applicant 1: base=0.520, contributions=[-0.028  0.095 -0.011]
Applicant 2: base=0.520, contributions=[ 0.077 -0.061  0.144]
```

Each SHAP value is that feature's push away from the average prediction
(`base`) for this one applicant, in the units of the model's output — a
large negative `debt_ratio` contribution for applicant 0 (-0.183) means
their debt ratio specifically pushed their approval probability down.
Summed with the base value and the other contributions, they reconstruct
the model's actual prediction for that row — this decomposition is what
makes SHAP suitable for individual "why did the model say this" reports,
not just aggregate summaries.

## Partial dependence: how does one feature affect predictions on average?

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib
matplotlib.use("Agg")
from sklearn.inspection import partial_dependence

pd_result = partial_dependence(forest, X, features=["debt_ratio"])
print(np.round(pd_result["grid_values"][0], 2))
print(np.round(pd_result["average"][0], 3))
```

```text
[0.01 0.09 0.18 0.27 0.36 0.45 0.53]
[0.72  0.68  0.6   0.5   0.39  0.29  0.21]
```

Partial dependence sweeps one feature across its range (holding others at
their observed distribution) and averages the model's predictions —
showing here that predicted approval probability falls steadily as debt
ratio rises, from ~72% at low debt to ~21% at high debt. This is the
"average effect" complement to SHAP's per-row explanations.

## Choosing the right tool

- **Need a single, portable, hand-interpretable number per feature**: use
  a linear/logistic model and read the coefficients.
- **Need a fast global ranking on a tree model**: impurity-based
  `feature_importances_`, treated as a first look, not gospel.
- **Need a trustworthy global ranking, model-agnostic**: permutation
  importance.
- **Need to explain one specific prediction to a person**: SHAP.
- **Need to see the shape of a feature's average effect**: partial
  dependence.

## Cheat sheet

| Tool | Answers |
|---|---|
| Linear coefficients | "How much and which direction, overall?" |
| `feature_importances_` | "What does the tree split on most?" (biased) |
| `permutation_importance` | "What does performance depend on?" (honest, any model) |
| SHAP | "Why did this one prediction come out this way?" |
| Partial dependence | "What's the average shape of this feature's effect?" |

## Exercise

Using the `loans` example, pick one applicant your `forest` model denies
(low predicted approval probability) and compute their SHAP values.
Write a two-sentence, plain-English explanation of the denial suitable
for a non-technical reader, based only on the sign and magnitude of their
per-feature contributions.
