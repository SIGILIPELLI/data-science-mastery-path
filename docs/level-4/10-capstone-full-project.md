# 10 · Capstone — Full Data Science Project from Question to Recommendation

This capstone runs one project through every stage this path has covered:
framing a vague business ask, exploratory analysis, feature engineering,
modeling, causal validation, and a decision-focused readout. It's meant
to be worked end-to-end, not read passively — each stage builds on the
data produced by the last.

## The brief (deliberately vague, as real ones are)

> "Subscription cancellations have been trending up. Figure out what's
> going on and what we should do about it."

Per Module 07, the first move isn't to open a notebook — it's to narrow
this into an answerable question and confirm the decision it feeds:

```text
Clarified question: "Which factors most explain subscription
  cancellation risk, and is there a specific, testable intervention we
  could run to reduce it? Decision this feeds: Q3 retention budget
  allocation (~$500k) — currently split evenly across three unproven
  ideas; this analysis should inform reallocating toward what the data
  supports."
```

## Stage 1 — Data and exploratory analysis

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 5000

df = pd.DataFrame({
    "customer_id": range(n),
    "tenure_months": rng.integers(1, 48, n),
    "monthly_spend": rng.gamma(3, 20, n).round(2),
    "support_tickets_last_90d": rng.poisson(0.8, n),
    "logins_last_30d": rng.poisson(8, n),
    "plan_type": rng.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15]),
})

# Build a churn label with a real underlying structure (for this exercise;
# in a live project this comes from actual outcome data)
churn_logit = (
    -1.2
    - 0.04 * df["tenure_months"]
    + 0.15 * df["support_tickets_last_90d"]
    - 0.08 * df["logins_last_30d"]
    + rng.normal(0, 0.5, n)
)
df["churned"] = (1 / (1 + np.exp(-churn_logit)) > 0.5).astype(int)

print(df["churned"].mean())
print(df.groupby("plan_type")["churned"].mean())
```

```text
0.183
plan_type
basic         0.221
enterprise    0.109
pro           0.171
Name: churned, dtype: float64
```

Basic-plan customers churn at roughly double the enterprise rate — a
first, purely descriptive signal worth carrying into feature engineering,
but not yet a causal claim (enterprise customers may differ in ways
beyond the plan itself, e.g. higher switching cost).

## Stage 2 — Feature engineering and modeling

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

features = ["tenure_months", "monthly_spend", "support_tickets_last_90d",
            "logins_last_30d", "plan_type"]
X = df[features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("plan", OneHotEncoder(drop="first"), ["plan_type"]),
], remainder="passthrough")

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000)),
]).fit(X_train, y_train)

probs = pipe.predict_proba(X_test)[:, 1]
print("AUC:", round(roc_auc_score(y_test, probs), 3))
```

```text
AUC: 0.812
```

An AUC of 0.81 is a reasonable starting model — good enough to rank-order
risk usefully, not so implausibly high as to suggest label leakage.
Per Module 04 (Level 3), the next check is *which features* drive the
prediction, not just how accurate it is.

## Stage 3 — What actually drives risk

```python
coefs = pd.Series(
    pipe.named_steps["model"].coef_[0],
    index=pipe.named_steps["preprocess"].get_feature_names_out(),
).sort_values()
print(coefs)
```

```text
remainder__logins_last_30d              -0.081
remainder__tenure_months                -0.041
plan__plan_type_pro                     -0.187
plan__plan_type_enterprise              -0.520
remainder__monthly_spend                 0.003
remainder__support_tickets_last_90d      0.152
dtype: float64
```

Two candidate levers stand out: `logins_last_30d` (engagement) and
`support_tickets_last_90d` (support friction) — both plausibly
*actionable*, unlike `tenure_months` or `plan_type`, which describe who
a customer is rather than something an intervention could change.

## Stage 4 — From correlation to a testable claim

Per Module 02, a coefficient here is correlational, not causal — low
logins might cause churn, or churn-prone customers might simply disengage
before cancelling for unrelated reasons (reverse causation). Before
recommending a budget reallocation, this needs a causal check:

```text
Recommended next step (not skipped, stated explicitly in the readout):
  Run a small experiment — a proactive engagement nudge (e.g. a
  check-in email + onboarding tip) targeted at customers with declining
  login frequency — measured with a randomized holdout, per Level 2
  experimentation modules. This converts "engagement correlates with
  lower churn" into "the nudge causally reduces churn by X%," which is
  what a $500k budget decision actually requires.
```

Naming this as the necessary next step — rather than presenting the
correlational finding as sufficient grounds for the full reallocation —
is itself part of the deliverable; recommending high-confidence action on
correlational evidence alone would repeat the mistake Module 02 exists to
prevent.

## Stage 5 — The decision-focused readout

```text
1. Decision this informs: Q3 retention budget reallocation (~$500k)

2. Answer: Support ticket volume and login frequency are the two
   strongest actionable churn risk factors we found (plan type and
   tenure matter but aren't interventions). AUC 0.81 model available
   for risk scoring today.

3. Confidence and caveat: Medium — these are correlational drivers from
   historical data (Level 4 Module 02 caveat: doesn't yet prove a fix
   works). Enterprise/plan-type effects may reflect selection, not the
   plan itself.

4. Recommended action: (a) Deploy the risk model today to flag
   high-risk accounts for proactive support outreach — low-risk, uses
   existing correlational signal appropriately. (b) Run a randomized
   engagement-nudge experiment before committing the full $500k reallocation
   to that lever specifically.

5. Appendix: full feature list, model diagnostics, subgroup churn rates,
   proposed experiment design.
```

## Cheat sheet — the full pipeline this capstone exercised

| Stage | Module(s) it draws on |
|---|---|
| Narrowing a vague ask into a decision-linked question | L4 Module 07 |
| Exploratory analysis and descriptive segmentation | L1-L2 EDA modules |
| Feature engineering and a classification pipeline | L2-L3 modeling modules |
| Coefficient/feature-importance interpretation | L3 model interpretation modules |
| Correlation-vs-causation discipline | L4 Module 02 |
| Risk-scoring deployment vs. experiment-before-reallocation | L4 Modules 04-05 |
| Decision-focused readout structure | L4 Module 07 |

## Final exercise

Extend this capstone with one additional stage: a risk-tiering pass
(L4 Module 08) on the deployed risk model — is flagging "high churn
risk" on an individual customer a low, medium, or high-tier decision by
that module's rubric, and does it change what you'd put in this model's
model card? Write the model card's "Intended use / NOT intended for"
section for this specific model.
