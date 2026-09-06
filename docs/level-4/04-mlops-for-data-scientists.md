# 04 · MLOps for Data Scientists

A model that scores 0.89 AUC in a notebook and is never deployed reliably
delivers 0.00 business value. MLOps is the set of practices that get a
model from "works in my notebook" to "runs correctly in production, and
someone finds out fast when it stops." This module covers the parts of
that pipeline a data scientist (not a dedicated ML engineer) is commonly
responsible for.

## The lifecycle beyond training

```text
Train → Validate → Package → Deploy → Monitor → Retrain → (repeat)
                                          ↑
                          this loop is what "MLOps" actually refers to —
                          training a good model is the easy 20%
```

Most data science curricula stop at "Validate." The remaining stages are
where models actually fail in practice: a model with excellent offline
metrics that's deployed via a manual, undocumented process, monitored not
at all, and never retriggered for retraining will silently degrade and
no one will notice until a stakeholder complains.

## Packaging: making a model reproducibly loadable

```python
import joblib
import json

# Save the model AND the exact preprocessing it expects
joblib.dump(model, "model_v3.pkl")

metadata = {
    "model_version": "v3",
    "trained_on": "2024-06-01",
    "feature_order": ["income", "tenure_months", "days_since_last_purchase"],
    "sklearn_version": "1.4.2",
    "training_metric": {"auc": 0.891},
}
with open("model_v3_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

```python
# At inference time — fail loudly on mismatch rather than silently
# feeding columns in the wrong order
def load_and_validate(model_path: str, metadata_path: str):
    model = joblib.load(model_path)
    with open(metadata_path) as f:
        meta = json.load(f)
    return model, meta

def predict(model, meta, features: dict):
    row = [features[col] for col in meta["feature_order"]]  # explicit order, not dict order
    return model.predict([row])[0]
```

The `feature_order` check is the single highest-value line in this
snippet: a model silently fed columns in the wrong order still produces a
prediction — it just produces a *wrong* one, with no error to signal it.

## Serving patterns

```text
Batch scoring:
  - Run predictions on a schedule (e.g. nightly) over a table of records
  - Simple, no latency requirement, easiest to get right first
  - Good default when "real-time" isn't actually a requirement

Online/real-time serving:
  - A request comes in, a prediction is returned within an SLA (e.g. 100ms)
  - Needs the online half of the feature store (Module 03)
  - Justified only when the product genuinely needs a live decision
```

```python
# Minimal real-time serving endpoint (FastAPI)
from fastapi import FastAPI
import joblib

app = FastAPI()
model, meta = load_and_validate("model_v3.pkl", "model_v3_metadata.json")

@app.post("/predict")
def predict_endpoint(features: dict):
    try:
        pred = predict(model, meta, features)
        return {"prediction": float(pred), "model_version": meta["model_version"]}
    except KeyError as e:
        return {"error": f"missing feature: {e}"}
```

Defaulting to batch scoring and only moving to real-time serving when a
concrete product requirement demands it avoids a large class of
unnecessary operational complexity — real-time serving adds a live
dependency, an SLA to meet, and a new failure surface that batch scoring
doesn't have.

## Monitoring: what to watch after deploy

```python
import numpy as np
from scipy.stats import ks_2samp

def feature_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.01) -> bool:
    """KS test: is the current feature distribution meaningfully different
    from what the model was trained on?"""
    stat, p_value = ks_2samp(reference, current)
    return p_value < alpha

training_income = np.random.normal(60000, 15000, 5000)
production_income_this_week = np.random.normal(63000, 15000, 500)
print(feature_drift(training_income, production_income_this_week))
```

```text
Monitor at three levels:
1. Operational  — latency, error rate, uptime (standard software monitoring)
2. Data         — input feature distributions vs. training (drift, above)
3. Model        — prediction distribution, and TRUE performance once
                  labels arrive (may be days/weeks delayed — churn,
                  fraud, and many labels aren't immediate)
```

The delayed-label problem is the part most new-to-MLOps data scientists
miss: for a churn model, you may not know whether a prediction was
correct for 30-90 days. Data-level drift monitoring is what gives an
early warning *before* the delayed ground truth confirms a problem.

## Retraining triggers

```text
Options, roughly in order of sophistication:
1. Fixed schedule (retrain monthly regardless of signal) — simple, wasteful if nothing changed
2. Performance-triggered (retrain when live metric drops below threshold) — needs fast-arriving labels
3. Drift-triggered (retrain when input distributions shift materially) — works even with delayed labels
4. Manual (a human decides) — fine at low model count, doesn't scale
```

A schedule-only strategy is a reasonable starting point, but pairing it
with a drift-triggered alert (option 3) catches the case where something
breaks *between* scheduled retrains — e.g. an upstream data pipeline
change that shifts a feature's distribution overnight.

## Cheat sheet

| Stage | Data scientist's responsibility |
|---|---|
| Packaging | Save model + exact preprocessing/feature order together |
| Serving choice | Default to batch; justify real-time with a concrete need |
| Operational monitoring | Usually owned by platform/infra, but know what's tracked |
| Data drift monitoring | Often the DS's job — you know what "normal" looks like |
| Retraining trigger | Define explicitly; don't leave it to "someone will notice" |

## How It Actually Works

**Why feature order silently corrupts predictions**: a fitted scikit-learn
model stores its learned coefficients (or split thresholds) positionally —
`model.coef_[0]` is "whatever the first training column was," with no
column-name awareness baked into the fitted object once it's serialized as
a plain array. Feeding `[tenure_months, income, days_since_last_purchase]`
at inference time when the model was trained on
`[income, tenure_months, days_since_last_purchase]` doesn't raise any
error — it just multiplies each coefficient against the wrong feature's
value, producing a numerically valid but meaningless prediction. This is
exactly the kind of bug that offline validation can't catch (validation
data uses the same code path, so it's consistently "wrong" in the same
consistent way) and that only shows up as unexplained production
degradation — which is why enforcing `feature_order` explicitly at
inference, rather than trusting dict ordering, converts a silent
correctness bug into a loud `KeyError`.

**The KS test for drift** works by comparing the two samples' empirical
CDFs directly: `ks_2samp` computes the maximum vertical distance between
the reference distribution's cumulative distribution function and the
current distribution's, `D = max|F_ref(x) - F_current(x)|`. Under the null
hypothesis that both samples come from the same underlying distribution,
`D`'s sampling distribution is known (it depends only on the two sample
sizes), which is what lets the test convert an observed `D` into a
p-value without assuming any particular distributional shape for the
feature itself — it's nonparametric, unlike a t-test's normality
assumption, which matters because feature distributions in production
(income, click counts) are routinely skewed.

**Why drift monitoring matters more than label-based monitoring for many
models**: true performance monitoring requires ground-truth labels, and for
churn, fraud, or LTV-style targets, the label only exists 30-90 days (or
more) after the prediction was made — a genuine reporting lag, not a
tooling gap. A model that starts silently misfiring on day 1 due to an
upstream schema change won't show up in a labeled performance metric until
day 30+, but a KS test on the affected feature's distribution can flag the
shift within the same monitoring cycle it happens in — drift is a
*leading* indicator, label-based performance is a *lagging* one, and
production monitoring needs both because drift without label deterioration
can also be a false alarm (a benign shift the model happens to be robust
to).

## Exercise

Take a model from an earlier module. Write the `metadata.json` it should
ship with (feature order, training date, key metric). Then write a single
`feature_drift` check for its most important input feature, and decide:
weekly, monthly, or on-drift-alert — which retraining trigger fits this
specific model's label latency, and why?
