---
description: "Data Storytelling for Executives — A technically correct analysis that doesn't change a decision has failed at its actual job. This module covers how to…"
---

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

## How It Actually Works

The **pyramid principle** (Barbara Minto's framework, from which this
module's structure is drawn) isn't just a formatting preference — it's
built around how working memory actually processes incoming information.
Working memory holds roughly 4±1 chunks of new information at once before
older chunks start getting displaced; a bottom-up narrative (data → method
→ analysis → conclusion) forces the listener to hold every intermediate
step in memory *while waiting* for the payoff, and any interruption (a
question, a distraction) before the conclusion arrives loses the thread
entirely. Leading with the conclusion converts the rest of the talk into
elaboration of something already understood — each supporting point is now
evaluated against a known claim rather than added to an unresolved stack,
which is dramatically lower cognitive load and survives interruption.

**The uncertainty-framing example** is a direct application of the
confidence-interval logic from Module 07/10 of Level 2: "$1.5M–$2.5M" is
functionally a stated interval width communicating the same information a
95% CI would (`estimate ± margin`), just in business language instead of
"a two-proportion z-test at α=0.05." The "falsely precise" failure mode
(a single point estimate with no stated uncertainty) is misleading for the
same statistical reason a naive point estimate without a confidence
interval is misleading in Level 2 — it implies a level of certainty the
underlying sample size and assumptions don't actually support. Naming the
assumption behind the number (the comparable program's 30% success rate)
is what lets a sharp stakeholder attack the assumption instead of
misreading the number as more solid than it is — the executive-communication
equivalent of showing your model's inputs, not just its output.

**Chart selection matching the claim** works because different chart types
encode different visual comparisons more or less efficiently for human
perception (the same perceptual hierarchy — position and length read faster
and more accurately than color or area — that Module 03's small-multiples
discussion relies on). A bar chart of `churn_rate` by `segment` puts the
comparison you're making (which segment churns more) directly onto length,
the single most accurately-perceived visual channel; a pie chart or a
stacked area chart would force the same comparison onto angle or stacked
position, both of which measurably increase perception error in controlled
studies. Choosing the chart is really choosing which visual channel the
key comparison gets mapped to.

## Exercise

Take the churn analysis from module 10 of Level 2 (the A/B test project)
and rewrite its recommendation as a one-page executive summary using the
template above. Keep it under 150 words total, and make sure the very
first sentence would survive as the only thing a busy VP reads.
