---
description: "Intro to Machine Learning for Data Science — Data scientists don't need to be ML researchers, but they do need to know how to frame a business question as…"
---

# 05 · Intro to Machine Learning for Data Science

Data scientists don't need to be ML researchers, but they do need to know
how to frame a business question as a supervised learning problem, fit a
baseline model, and judge whether it's actually useful. This module covers
that workflow end to end with scikit-learn.

## Framing the problem

Every supervised ML problem needs three things: a **target** (what you're
predicting), a set of **features** (what you predict it from), and a
**metric** (how you'll know if predictions are good). Get these wrong and
no amount of modeling saves you — e.g. predicting "will this customer
churn next month" needs features known *before* the churn happens, not
features computed from data only available after (label leakage).

```python
import pandas as pd
import numpy as np

np.random.seed(0)
n = 500
customers = pd.DataFrame({
    "tenure_months": np.random.randint(1, 60, n),
    "monthly_spend": np.random.gamma(4, 20, n),
    "support_tickets": np.random.poisson(1.2, n),
})
# Simulate a churn label correlated with tenure and tickets
logit = -0.03 * customers["tenure_months"] + 0.4 * customers["support_tickets"] - 1.0
prob = 1 / (1 + np.exp(-logit))
customers["churned"] = (np.random.rand(n) < prob).astype(int)
print(customers["churned"].value_counts(normalize=True).round(2))
```

```text
0    0.71
1    0.29
```

## Train/test split

Never evaluate a model on the data it was trained on — it will look better
than it really is. Hold out a test set the model never sees during fitting.

```python
from sklearn.model_selection import train_test_split

X = customers[["tenure_months", "monthly_spend", "support_tickets"]]
y = customers["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)
```

```text
(375, 3) (125, 3)
```

`stratify=y` keeps the churn rate roughly equal in both splits — important
for an imbalanced target like this one (~29% positive class).

## A baseline model: logistic regression

Always fit the simplest reasonable model first. It's a sanity check, and
often it's competitive.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

baseline = make_pipeline(StandardScaler(), LogisticRegression())
baseline.fit(X_train, y_train)
print(baseline.score(X_test, y_test).__round__(3))
```

```text
0.784
```

Wrapping the scaler and the model in a `Pipeline` guarantees the same
transformation is applied consistently to train and test data, and it
prevents a common mistake: fitting the scaler on the full dataset (which
leaks test-set statistics into training).

## A stronger model: random forest

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
forest.fit(X_train, y_train)
print(forest.score(X_test, y_test).__round__(3))
```

```text
0.808
```

Random forests handle nonlinear relationships and feature interactions
without manual feature engineering, and they rarely need scaling. But
accuracy alone can be misleading on imbalanced data — a model that
predicts "never churns" would score ~71% here without learning anything.

## Beyond accuracy: precision, recall, and the confusion matrix

```python
from sklearn.metrics import classification_report, confusion_matrix

preds = forest.predict(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, digits=2))
```

```text
[[82  7]
 [17 19]]
              precision    recall  f1-score   support
           0       0.83      0.92      0.87        89
           1       0.73      0.53      0.61        36
```

Recall on the churn class (0.53) says the model only catches about half of
actual churners — if the business cost of missing a churner is high, this
model isn't good enough yet, regardless of its 81% accuracy. Which metric
matters is a business decision, not a modeling one: state it before you
start comparing models.

## Feature importance as a sanity check

```python
importances = pd.Series(forest.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).round(3))
```

```text
tenure_months      0.52
support_tickets    0.34
monthly_spend      0.14
```

This matches the simulation: tenure and support tickets drove the label,
spend didn't. In real projects, an importance ranking that doesn't match
domain intuition is a signal to check for leakage or a data bug before you
trust the model at all.

## Cheat sheet

| Task | Code |
|---|---|
| Split data | `train_test_split(X, y, test_size=.25, stratify=y)` |
| Scale + model in one object | `make_pipeline(StandardScaler(), LogisticRegression())` |
| Fit / predict | `model.fit(X_train, y_train)`, `model.predict(X_test)` |
| Precision/recall/F1 | `classification_report(y_test, preds)` |
| Feature importance | `model.feature_importances_` (tree models) |

## How It Actually Works

**Logistic regression** doesn't predict 0/1 directly — it fits a linear
combination `z = β₀ + β₁x₁ + ... + βₙxₙ` and squashes it through the sigmoid
`p = 1 / (1 + e⁻ᶻ)`, which maps any real number to `(0, 1)`. The coefficients
`β` are found by maximizing the likelihood of the observed labels — no
closed-form solution exists (unlike linear regression's normal equation), so
solvers use iterative gradient-based optimization on the log-loss
`-Σ [y·log(p) + (1-y)·log(1-p)]`, which penalizes confident-and-wrong
predictions far more steeply than log-loss penalizes uncertainty. The
decision boundary (`p = 0.5`) is a straight line/hyperplane in feature space
— which is exactly why it's called a *linear* model and why it struggles
when the true boundary between classes is curved.

**Random forest** fixes a single decision tree's biggest weakness — high
variance, since a tree fit to one bootstrap sample of the data can look very
different from a tree fit to another — by averaging many trees. Each tree is
trained on a bootstrap resample (drawing `n` rows with replacement, so each
tree sees a slightly different dataset), and at every split each tree is
only allowed to consider a random subset of features (typically `√p` of
them for classification). Averaging predictions across many trees that are
individually noisy but decorrelated from each other (because of the row and
feature randomness) reduces variance without much increasing bias — an
instance of the general "wisdom of crowds" fact that averaging `k`
uncorrelated estimators reduces variance by roughly a factor of `k`.

**Precision, recall, and the confusion matrix** exist because "accuracy"
collapses two different error types into one number. Given actual positives
and negatives crossed with predicted positives and negatives, you get four
counts: true positives (TP), false positives (FP), true negatives (TN), and
false negatives (FN). `Precision = TP / (TP + FP)` answers "of everything I
flagged positive, how much was right?" — it's what you care about when a
false alarm is costly. `Recall = TP / (TP + FN)` answers "of everything that
was actually positive, how much did I catch?" — what matters when a missed
case is costly (fraud, disease). On an imbalanced dataset (say 95% negative
class), a model that always predicts negative gets 95% accuracy while
having zero recall — which is precisely the failure mode precision/recall
exist to expose.

**Feature importance** from a tree ensemble is computed by summing, over
every split in every tree that uses a given feature, how much that split
reduced impurity (Gini impurity or entropy), weighted by how many samples
passed through that split. A feature used near the root of many trees on
large sample counts accumulates a high score; a feature never selected for
any split scores zero. This is a measure of *how much the model relied on
it*, not proof of a true causal or even correlational relationship in the
real world — a leaked feature (one that encodes the answer, like a
post-outcome timestamp) will show up with suspiciously high importance
precisely because the model exploited it, which is why importance rankings
double as a leakage detector.

## Exercise

Using the `customers` DataFrame above, add a fourth feature — say,
`avg_ticket_priority` (simulate it) — and refit both models. Report
whether recall on the churn class improved, and explain in one paragraph
why accuracy alone would have hidden that answer.
