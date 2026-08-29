# 01 · What Is Data Science?

Data science is the practice of turning raw, messy data into decisions. It
sits at the intersection of statistics (does this pattern mean anything?),
programming (can I actually process this data at scale?), and domain
knowledge (does this finding make sense, and what should we do about it?).
This module maps the role, the day-to-day workflow, and how data science
differs from the two roles it's most often confused with — data engineering
and machine learning engineering.

## The core workflow

Almost every data science project — from a one-off analysis to a shipped
model — follows the same loop:

```text
1. Ask a question       "Why did signups drop 12% last month?"
2. Get the data         pull from a warehouse, API, CSV export, logs
3. Clean it             fix types, handle missing values, remove duplicates
4. Explore it (EDA)     summary stats, distributions, correlations, plots
5. Analyze / model it   hypothesis tests, regression, or an ML model
6. Communicate it       a report, dashboard, or recommendation
7. Decide / act         someone uses the finding to change something
```

Steps 2–4 routinely eat 60–80% of real project time — cleaning and
understanding data, not building models, is the actual day job. This track
spends Levels 1–2 making sure that majority of the work is solid before
touching anything resembling machine learning.

A minimal but real example of the loop, end to end:

```python
import numpy as np
import pandas as pd

# 1. Ask: did the new checkout flow change average order value?
# 2. Get: (synthetic order data standing in for a warehouse pull)
rng = np.random.default_rng(7)
orders = pd.DataFrame({
    "flow": ["old"] * 500 + ["new"] * 500,
    "order_value": np.concatenate([
        rng.normal(58, 15, 500),
        rng.normal(62, 15, 500),
    ]),
})

# 3. Clean: no missing values here, but always check
print(orders.isna().sum().sum())          # 0

# 4. Explore
print(orders.groupby("flow")["order_value"].mean().round(2))

# 5. Analyze (a proper hypothesis test comes in Module 06)
diff = orders.groupby("flow")["order_value"].mean()
print(f"difference: {diff['new'] - diff['old']:.2f}")
```

```text
0
flow
new    61.76
old    56.08
Name: order_value, dtype: float64
difference: 5.68
```

Notice step 6 is missing from the code — "the new flow raised average order
value by $5.68" is a sentence, not a plot. Writing that sentence, with the
right caveats, is as much a part of data science as the `groupby`.

## Data science vs. data engineering vs. ML engineering

These three roles get merged in job postings but do genuinely different
work. Knowing the boundary matters because it tells you who to ask for what.

| | Data Engineering | Data Science | ML Engineering |
|---|---|---|---|
| **Primary output** | Reliable pipelines & tables | Insights, reports, decisions | Deployed, maintained ML models |
| **Typical question** | "How do we get this data flowing reliably?" | "What does this data tell us, and what should we do?" | "How do we serve this model at scale and keep it working?" |
| **Core tools** | Airflow, Spark, dbt, warehouses | pandas, SQL, stats, notebooks, BI tools | Docker, Kubernetes, model registries, monitoring |
| **Success looks like** | Data lands on time, uncorrupted | A correct, well-communicated recommendation | Low latency, high uptime, no silent model decay |
| **Failure mode** | Broken/late pipeline, silent data loss | A wrong conclusion presented with false confidence | A model that quietly degrades and nobody notices |

In practice the roles overlap heavily at smaller companies — a data
scientist often writes their own SQL pipelines (light data engineering) and
ships a model to production (light ML engineering). This track's Level 1–2
builds the statistics/EDA/communication core that is uniquely "data
science"; Level 2's "Intro to ML for Data Science" and Level 4's "MLOps for
Data Scientists" modules deliberately border on the neighboring roles so you
know where your work ends and a hand-off begins.

## What "good" data science looks like

A useful gut-check for any analysis, revisited throughout this track:

- **Reproducible** — someone else (including future you) can re-run it and
  get the same answer.
- **Honest about uncertainty** — "sales are up" vs. "sales are up 4%, but
  that's within the normal week-to-week noise" are very different claims.
  Module 06 builds the statistical vocabulary for this.
- **Scoped to a decision** — an analysis that doesn't change what anyone
  does is trivia, not data science. Always trace back to "so what should we
  do differently?"
- **Visually honest** — a chart can make a 1% difference look enormous by
  truncating an axis. Module 09 covers this directly.

## Cheat sheet

| Concept | One-line definition |
|---|---|
| Data science | Turning raw data into a decision, using stats + code + domain sense |
| Data engineering | Building the pipelines that make data reliably available |
| ML engineering | Deploying and operating models in production at scale |
| EDA | Exploratory Data Analysis — looking at data before modeling it |
| The 80% rule | Cleaning/understanding data usually dominates project time |
| "So what?" test | Every finding should trace to a decision or action |

## Exercise

Pick any statistic you've seen in the news this week (a headline like "X
rose by Y%"). Write three sentences: (1) what question was the analyst
originally trying to answer, (2) what data they most likely needed to answer
it, and (3) one way the number could be misleading without more context
(sample size, time window, comparison baseline). You'll build the tools to
formally catch these issues starting in Module 06.
