# 06 · Data Science Strategy & Roadmapping

A backlog of interesting projects isn't a strategy. This module covers
how to build a data science roadmap that ties directly to business
priorities, defensibly sequences work, and survives contact with a
budget conversation.

## Start from business outcomes, not techniques

```text
Weak roadmap item:  "Build a causal inference framework"
Strong roadmap item: "Determine whether the loyalty program actually
                      drives retention, to inform a $2M/year renewal
                      decision" (which happens to require causal methods)
```

The weak version optimizes for what's technically interesting; the strong
version starts from a decision someone needs to make, with a dollar
figure or comparable stakes attached, and only then asks what technique
answers it. A roadmap built from technique-first items is difficult to
defend in a prioritization conversation because it has no natural way to
compare against a competing item that's expressed the same way.

## A simple strategy-to-roadmap chain

```text
1. Company strategic priority   (e.g. "improve retention this year")
        ↓
2. Business questions this raises  (e.g. "why do customers churn?
                                     which interventions reduce it?
                                     which segment matters most?")
        ↓
3. Data science initiatives that answer them  (churn driver analysis,
                                                intervention experiment,
                                                segment-level LTV model)
        ↓
4. Sequenced roadmap with dependencies  (driver analysis BEFORE
                                          intervention design, since the
                                          intervention should target
                                          the drivers found in step 3)
```

Each roadmap item should be traceable back up this chain to a company
priority in one or two hops. An item that can't be traced back — however
technically interesting — is a candidate for explicit deprioritization,
not because it's bad work, but because it's optimizing for something the
org isn't currently prioritizing.

## Sequencing: what must come before what

```python
import pandas as pd

initiatives = pd.DataFrame({
    "initiative": ["Churn driver analysis", "Retention intervention experiment",
                    "Segment LTV model", "Real-time churn scoring"],
    "depends_on": [None, "Churn driver analysis", None, "Churn driver analysis"],
    "quarter": ["Q1", "Q2", "Q1", "Q3"],
})
print(initiatives)
```

```text
                        initiative                    depends_on quarter
0           Churn driver analysis                          None      Q1
1  Retention intervention experiment        Churn driver analysis      Q2
2                  Segment LTV model                          None      Q1
3            Real-time churn scoring        Churn driver analysis      Q3
```

Making dependencies explicit (even in a simple table like this) surfaces
sequencing mistakes before they cost a quarter — e.g. building real-time
churn scoring before knowing which features actually drive churn risks
building infrastructure around the wrong signal.

## Communicating a roadmap upward

```text
For an exec audience, lead with the business question and expected
decision impact, not the method:

  "Q1: Understand what drives churn, so Q2's retention budget goes to the
   right intervention instead of being split evenly across guesses.
   Confidence: medium — first pass on 18 months of data.
   Risk: driver analysis is correlational; may need a follow-up
   experiment (Q2) to confirm causality before committing full budget."
```

Naming the *risk* and *confidence level* explicitly (rather than
presenting the plan as certain) is what earns credibility for the next
roadmap — a lead who has once said "medium confidence, here's why" and
been right is trusted more on the next "high confidence" claim than one
who has always presented plans with uniform certainty.

## Resourcing tradeoffs

| Lever | Effect | Cost |
|---|---|---|
| Add headcount to an initiative | Faster delivery, up to a point | Ramp-up time; coordination overhead grows non-linearly |
| Narrow initiative scope | Faster delivery of a smaller answer | May not fully answer the business question |
| Reuse existing platform tooling (Module 03) | Faster delivery, more reliable | Requires platform investment to already exist |
| Extend timeline | No added cost | Delays the business decision it informs |

A common strategy mistake is defaulting to "add headcount" as the lever
for every schedule pressure — coordination overhead among data scientists
on a tightly coupled analysis often grows faster than the added capacity,
so scope reduction or reuse is frequently the better first lever to pull.

## Revisiting the roadmap

```text
Quarterly checkpoint questions:
1. Did any Q1 finding change what Q2/Q3 should prioritize?
2. Did a business priority shift underneath the roadmap?
3. Is any "in progress" initiative still traceable to a live business question?
```

A roadmap that never changes after new findings arrive isn't being
informed by the work it commissions — a churn driver analysis that finds
the real driver is onboarding friction, not the price plan originally
suspected, should visibly redirect Q2's intervention work, not get
filed away while the original Q2 plan proceeds unchanged.

## Cheat sheet

| Practice | Prevents |
|---|---|
| Business-question-first framing | Technique-driven roadmaps no one can prioritize against |
| Explicit dependency chain | Building downstream work before its prerequisite |
| Confidence + risk stated upward | Overpromising, and losing credibility when reality diverges |
| Resourcing tradeoff table | Reflexively adding headcount as the only lever |
| Quarterly re-check against findings | A roadmap that ignores its own results |

## Exercise

Take a company priority (real or invented, e.g. "reduce customer support
cost"). Write the four-step chain (priority → questions → initiatives →
sequenced roadmap) down to at least three initiatives with explicit
dependencies, and one paragraph of upward communication in the
confidence/risk style shown above.
