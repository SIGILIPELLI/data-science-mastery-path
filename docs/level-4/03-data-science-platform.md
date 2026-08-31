# 03 · Building a Data Science Platform

Once an org has more than a handful of data scientists, the same problems
get solved independently in every pod: feature computation, experiment
tracking, model serving, scheduled jobs. A data science **platform** is
the shared infrastructure that turns those from N reinvented wheels into
one built well — this module covers what belongs in it, and what doesn't.

## What a platform is (and isn't)

```text
Platform SHOULD own:
  - Feature store (compute once, reuse across models)
  - Experiment tracking (params, metrics, artifacts — one place to look)
  - Model registry + serving infrastructure
  - Scheduled job orchestration
  - Shared compute (notebooks, training jobs) with sane defaults
  - Data access + governance (who can query what, PII handling)

Platform should NOT own:
  - Which model architecture a team chooses
  - Business logic specific to one team's problem
  - The actual analysis / decision-making
```

The failure mode in both directions is common: a platform team that tries
to own modeling decisions becomes a bottleneck and a target of resentment;
a platform team that only owns compute provisioning leaves every team
re-solving feature computation and experiment tracking from scratch.

## Feature store: the core abstraction

```python
# Without a feature store: every model's training script
# recomputes "days_since_last_purchase" slightly differently,
# and inference computes it a THIRD way — training/serving skew.

# With a feature store: one definition, used everywhere
from datetime import date

def days_since_last_purchase(customer_id: str, as_of: date) -> int:
    """Single source of truth — used by both training pipelines and
    the real-time serving path, so the feature can never silently diverge."""
    last_purchase = lookup_last_purchase(customer_id, as_of)
    return (as_of - last_purchase).days if last_purchase else -1
```

```text
Feature store responsibilities:
1. Offline store — historical feature values for training (point-in-time correct)
2. Online store  — low-latency lookup for real-time inference
3. Registry      — feature definitions, owners, freshness SLAs
4. Consistency   — guarantee the SAME transformation logic produces both
```

The single most valuable property of a feature store is closing
**training/serving skew** — the bug class where a feature is computed one
way offline (in a notebook or batch job) and subtly differently online (in
a low-latency service), producing a model that performs well in
validation and poorly in production for reasons that don't show up in any
offline metric.

## Experiment tracking as organizational memory

```python
import mlflow

mlflow.set_experiment("churn-model")

with mlflow.start_run(run_name="xgb-v3-tuned"):
    mlflow.log_params({"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300})
    # ... train model ...
    mlflow.log_metrics({"auc": 0.891, "precision_at_10pct": 0.62})
    mlflow.sklearn.log_model(model, "model")
```

The value compounds over time, not per-run: without shared tracking, "did
we already try a deeper tree for this problem?" requires asking around
Slack; with it, it's a filtered query. A platform's job is making the
logging call as close to zero-friction as `model.fit(...)` itself, or
adoption stalls.

## Model registry and promotion stages

```text
Stage: None → Staging → Production → Archived

mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="churn-model",
)
# Promotion is a deliberate, auditable action — not implicit from training
client.transition_model_version_stage(
    name="churn-model", version=3, stage="Production",
)
```

A registry with explicit stages (rather than "whichever model.pkl is in
the deploy folder") gives you two things a folder can't: an audit trail of
what was in production when, and a mechanical rollback path (`transition
... stage="Archived"`, promote the prior version) when a new model
underperforms after deploy.

## Orchestration: scheduled pipelines

```python
# Airflow-style DAG definition (conceptual)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG("daily_feature_refresh", start_date=datetime(2024, 1, 1),
         schedule="0 3 * * *", catchup=False) as dag:

    refresh_features = PythonOperator(
        task_id="refresh_customer_features",
        python_callable=refresh_customer_features,
    )
    retrain_check = PythonOperator(
        task_id="check_retrain_trigger",
        python_callable=check_retrain_trigger,
    )
    refresh_features >> retrain_check
```

Orchestration is where "someone remembers to run this notebook every
Monday" becomes "this runs at 3am, alerts on failure, and has a visible
run history" — the platform-level shift from individual diligence to
infrastructure that doesn't depend on anyone remembering anything.

## Build vs. buy

| Component | Build in-house when | Buy/adopt OSS when |
|---|---|---|
| Feature store | Very specific latency/scale needs | Feast, Tecton cover most needs |
| Experiment tracking | Rarely worth building | MLflow, Weights & Biases |
| Orchestration | Rarely worth building | Airflow, Dagster, Prefect |
| Model serving | Custom latency/hardware constraints | Seldon, BentoML, cloud-native endpoints |
| Governance/access | Usually org-specific | Often needs custom layer over existing IAM |

The default should be adopting an existing tool and investing engineering
effort in the *integration* (making it the path of least resistance for
your specific teams) rather than in the underlying tool — most platform
teams that build a bespoke feature store from scratch are solving a
problem several open-source projects already solved.

## Cheat sheet

| Component | Solves |
|---|---|
| Feature store | Training/serving skew, duplicated feature logic |
| Experiment tracking | "What have we already tried?" institutional memory |
| Model registry | Auditable promotion and rollback |
| Orchestration | Scheduled pipelines that don't depend on human memory |
| Governance layer | Who can access what data, PII handling |

## Exercise

Pick a model you've built in an earlier module. Sketch (in a text diagram)
what a minimal feature store entry for its top 3 features would look like:
the transformation logic, an offline (training) computation path, and an
online (serving) computation path. Identify any feature where those two
paths could plausibly diverge today.
