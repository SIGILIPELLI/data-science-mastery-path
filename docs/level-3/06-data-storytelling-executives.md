# 06 · Data Storytelling for Executives

A technically correct analysis that doesn't change a decision has failed
at its actual job. This module covers how to structure and present
findings for an executive audience: people with five minutes, no interest
in your methodology, and a decision to make.

## Lead with the answer, not the process

```text
# Weak opening (methodology-first)
"We pulled 18 months of transaction data, cleaned duplicate records,
segmented customers into quintiles by spend, and ran a churn model with
an AUC of 0.81..."

# Strong opening (answer-first)
"Our top 20% of customers by spend generate 68% of revenue but have a
churn rate 3x higher than the rest of the base. Retaining just half of
the churn-risk customers in that segment would protect roughly $2.1M in
annual revenue."
```

Executives read the first sentence and decide whether to keep reading.
Put the business conclusion and its size (dollars, percentage, headcount)
in that first sentence — methodology belongs in an appendix or a follow-up
question, not the opener.

## Structure: the pyramid principle

Organize top-down: **conclusion → key supporting points → supporting
detail**, not bottom-up the way you actually did the analysis.

```text
1. Headline: what should we do, and what's at stake
2. 2–4 supporting findings (each answerable as its own mini "so what")
3. Detail / methodology / caveats — only if asked, or in an appendix
```

This is the reverse of a lab report or academic paper, which builds up to
a conclusion. An executive's attention is a scarce resource being spent on
your slide; assume they may only read the headline and the first
supporting point.

## Choose the chart that matches the claim

```python
import pandas as pd

# The claim: "churn risk is concentrated in the top spend segment"
segments = pd.DataFrame({
    "segment": ["Top 20%", "Next 30%", "Bottom 50%"],
    "revenue_share": [0.68, 0.24, 0.08],
    "churn_rate": [0.18, 0.07, 0.05],
})
print(segments)
```

```text
      segment  revenue_share  churn_rate
0     Top 20%           0.68        0.18
1    Next 30%           0.24        0.07
2  Bottom 50%           0.08        0.05
```

A single well-labeled bar chart of `churn_rate` by `segment`, annotated
with the revenue share, makes the concentration obvious at a glance. Avoid
the common failure mode of putting *all* your exploratory charts in the
deck — pick the one or two that carry the argument, and cut the rest
(or move them to an appendix "for reference").

## Quantify uncertainty without hedging into meaninglessness

```text
# Too vague to act on
"Retention efforts might help somewhat."

# Falsely precise
"Retention efforts will save exactly $2,147,382."

# Right level of precision for a decision
"Based on a similar retention program's 30% success rate, we estimate
$1.5M–$2.5M in protected revenue if we target the top-spend, high-risk
segment. We recommend a pilot to validate the assumption before a
full rollout."
```

Executives need enough precision to compare options, not false confidence.
Naming the assumption behind the number (a comparable program's 30% rate)
lets a sharp stakeholder challenge the right thing instead of the number
itself.

## Anticipate the three questions you'll always get

1. **"How confident are we?"** — have the confidence interval or a
   sensitivity range ready, in the same units as your headline number.
2. **"What does it cost to act on this?"** — even a rough cost estimate
   ("a pilot costs ~2 engineer-weeks") turns your finding into an
   actionable trade-off instead of an abstract insight.
2. **"What happens if we do nothing?"** — frame the cost of inaction
   explicitly; it's often the most persuasive number in the deck.

## A one-page executive summary template

```text
HEADLINE
One sentence: the finding and its business size.

SO WHAT
2–3 bullets: why this matters now, tied to a goal the audience already cares about.

RECOMMENDATION
One clear ask: a decision, a resource, an approval — not "further study."

RISK / UNCERTAINTY
One line naming the biggest assumption or caveat, honestly.

APPENDIX (separate page/slide)
Methodology, full charts, data sources, statistical detail.
```

Sticking to a fixed template forces discipline — it's much harder to bury
a weak recommendation behind ten slides of charts when the format only
gives you one page to make the case.

## Cheat sheet

| Instinct to resist | Do instead |
|---|---|
| Lead with methodology | Lead with the conclusion and its size |
| Show every chart you made | Show the one or two that carry the argument |
| Hedge with vague language | Give a concrete range tied to a stated assumption |
| End with "more research needed" | End with a specific, actionable recommendation |

## Exercise

Take the churn analysis from module 10 of Level 2 (the A/B test project)
and rewrite its recommendation as a one-page executive summary using the
template above. Keep it under 150 words total, and make sure the very
first sentence would survive as the only thing a busy VP reads.
