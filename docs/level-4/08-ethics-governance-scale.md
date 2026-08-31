# 08 · Ethics & Governance at Scale

Module 09 in Level 3 covered auditing a single model for bias. At the
organizational level, the question changes: how do you make sure *every*
model that touches a real decision gets that kind of scrutiny, without
a manual review of each one becoming a bottleneck? This module covers
governance structures that scale ethics review across many models and
teams.

## Why ad-hoc review doesn't scale

```text
At 1 model:  the team that built it reviews it carefully. Works fine.
At 50 models across 10 teams: 
  - No consistent bar for what "reviewed" means
  - High-stakes models (credit, hiring, healthcare) get the same
    scrutiny as low-stakes ones (a recommendation widget) — or worse,
    less, because the high-stakes team is under more delivery pressure
  - No one has visibility into which models are even in production
```

The core governance failure at scale isn't usually malice — it's that
without a structural forcing function, review effort correlates with
team culture and time pressure rather than with actual stakes, which is
backwards.

## Risk-tiering models

```python
import pandas as pd

models = pd.DataFrame({
    "model": ["Product recommender", "Credit risk scoring", "Resume screening",
              "Ad click prediction", "Loan pricing"],
    "affects_legally_protected_class": [False, True, True, False, True],
    "reversible_if_wrong": [True, False, False, True, False],
    "individual_level_decision": [False, True, True, False, True],
})

def risk_tier(row) -> str:
    high_risk_flags = sum([
        row["affects_legally_protected_class"],
        not row["reversible_if_wrong"],
        row["individual_level_decision"],
    ])
    if high_risk_flags >= 2:
        return "high"
    elif high_risk_flags == 1:
        return "medium"
    return "low"

models["tier"] = models.apply(risk_tier, axis=1)
print(models[["model", "tier"]])
```

```text
                 model    tier
0  Product recommender     low
1  Credit risk scoring    high
2     Resume screening    high
3   Ad click prediction     low
4        Loan pricing     high
```

A tiering rubric like this turns "which models need a full fairness
audit" from a subjective, team-by-team judgment call into a mechanical
classification applied consistently — high-tier models get mandatory
review before launch; low-tier models get a lighter self-certified
checklist, keeping review effort proportional to actual stakes rather
than to which team happens to be conscientious.

## A model card as the governance artifact

```text
model_card.md — required before a high-tier model reaches production

## Model: Loan Pricing v4
- Intended use: pricing for personal loans $1k-$25k, US applicants
- NOT intended for: business loans, non-US applicants, auto-decline
  decisions without human review

## Training data
- Source, date range, size
- Known gaps: under-represents applicants with < 6 months credit history

## Fairness evaluation
- Metrics used: equal opportunity, selection rate parity — see Module 09
- Groups evaluated: [protected attributes tested]
- Results: [gap sizes, mitigation applied if any]

## Performance
- Overall AUC / calibration
- Performance BY SUBGROUP, not just overall (a model can look fine in
  aggregate metrics while performing worse for a specific subgroup)

## Monitoring plan
- What triggers a re-review (drift threshold, complaint volume, time-based)

## Approved by
- [name, role, date] — the accountability record
```

The subgroup performance section is the one most often skipped and most
important: an aggregate AUC of 0.85 can hide a model that performs at
0.70 for a smaller subgroup — a gap invisible in the headline metric but
directly relevant to whether the model should ship as-is.

## Governance structure: who approves what

```text
Low tier:   Self-certified checklist, no external review required
Medium tier: Peer review from another DS team + model card
High tier:  Model card + fairness audit + sign-off from a designated
            review body (ethics committee, legal, or a senior
            cross-functional group depending on org size) BEFORE launch
```

Attaching the review requirement to launch (a gate, not a suggestion) is
what makes it actually happen under deadline pressure — a review step
that's "recommended" rather than blocking is the first thing cut when a
launch date is at risk, which is precisely when review matters most.

## Ongoing governance, not just launch review

```text
A model approved at launch can still drift into a governance problem:
- Retraining on new data can silently reintroduce a bias that was
  mitigated in the original version — re-run the fairness audit on
  every retrain of a high-tier model, not just the first launch
- A model built for one use case gets repurposed for another without
  re-review ("we already have a risk score, let's use it for X too")
- Complaint/appeal data (if the decision has an appeals process) is a
  monitoring signal governance should actually look at, not just file
```

The "repurposing without re-review" failure mode is common precisely
because it looks efficient in the moment — reusing an existing risk
score for a new decision context feels like avoiding duplicated work,
but the original model card's "intended use" section exists specifically
to flag that the fairness evaluation doesn't automatically transfer to a
new use case.

## Cheat sheet

| Practice | Prevents |
|---|---|
| Risk tiering rubric | Review effort misallocated relative to actual stakes |
| Mandatory model card (high tier) | Undocumented intended use, gaps, and evaluation |
| Subgroup performance reporting | Aggregate metrics hiding a poorly-served subgroup |
| Launch gate (not a suggestion) | Review being cut under deadline pressure |
| Re-audit on retrain | Bias silently reintroduced by a retraining cycle |
| Re-review on repurposing | A model's fairness evaluation not transferring to a new use case |

## Exercise

Apply the risk-tiering rubric to three models (real or hypothetical) from
your own work or an earlier module. For the highest-tier one, draft the
"Intended use / NOT intended for" and "Monitoring plan" sections of a
model card in full.
