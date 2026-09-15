---
description: "Cross-Functional Stakeholder Management — Technically correct analysis that never gets acted on has failed at its actual job. This module covers the…"
---

# 07 · Cross-Functional Stakeholder Management

Technically correct analysis that never gets acted on has failed at its
actual job. This module covers the non-technical skill that most
determines whether a senior data scientist's work drives decisions:
managing the relationship with the stakeholders who'll act (or not) on
it.

## Diagnosing the stakeholder before the analysis

```text
Questions to answer BEFORE starting the analysis, not after delivering it:
1. Who makes the actual decision this informs?
2. What decision are they trying to make? (often narrower than what was asked)
3. What would change their mind, specifically?
4. What's their default action if I deliver nothing? (the real baseline
   you're competing against — not "no decision," usually "whatever they
   were already going to do")
5. What's their math/stats fluency, and how should that shape the
   final deliverable's form?
```

A request phrased as "can you look into why revenue dropped in March"
often has a much narrower real decision behind it ("should we pause the
Q2 pricing change we already have planned") — surfacing that in a
clarifying conversation up front avoids delivering a broad root-cause
analysis when a narrow yes/no answer was what was actually needed.

## Translating a vague ask into an answerable question

```text
Vague ask:       "Can you look into customer churn?"
Clarifying Qs:   "Is this about understanding causes, predicting who's
                  at risk, or evaluating whether a specific program
                  works? Is there a decision this feeds — budget,
                  staffing, a go/no-go on a program?"
Answerable form: "Which of our three current retention levers has the
                  best cost-per-retained-customer, so we can reallocate
                  the Q3 budget toward it?"
```

Skipping the clarifying step is the single most common cause of "good
analysis, wrong question" — a technically sound answer to the vague
version of the ask can be entirely unusable for the actual decision
behind it, and that mismatch is often invisible to the stakeholder too
until they see the delivered analysis.

## Managing disagreement about the result

```text
When a stakeholder pushes back on a finding they don't like:

Don't:  Restate the same numbers louder, or defer entirely to their
        intuition and drop a finding you believe is correct.

Do:     1. Separate "you're wrong about the data" from "you disagree
           about what to do given the data" — these need different
           responses.
        2. Ask what evidence WOULD change their view — often surfaces
           an unstated assumption or a legitimate blind spot in the
           analysis worth checking.
        3. If the disagreement survives that, present both the finding
           and the disagreement transparently to whoever makes the
           final call, rather than quietly softening the finding.
```

The instinct to soften a result under pushback (rounding a jarring number
down, adding hedges that weren't in the original analysis) to keep the
relationship comfortable is worth naming explicitly as a failure mode —
it trades short-term social ease for long-term credibility, and stakeholders
generally notice a pattern of analysis conveniently aligning with what
they wanted to hear.

## Managing expectations on uncertain results

```python
import numpy as np

point_estimate = 0.034     # 3.4% lift
ci_low, ci_high = -0.008, 0.076   # 90% CI includes zero
```

```text
Weak framing: "The new onboarding flow increased conversion by 3.4%."
Strong framing: "We estimate a 3.4% lift, but the 90% confidence interval
                 (-0.8% to +7.6%) includes zero — we can't rule out no
                 effect. Given the sample size, I'd recommend [running
                 longer / treating this as directional only / a specific
                 next step] before committing budget on the strength of
                 this number alone."
```

Presenting a point estimate without its uncertainty to a stakeholder who
then makes an irreversible budget decision on it is a common way rigorous
analysis produces a bad outcome anyway — the fix is making the
uncertainty and its practical implication for the decision explicit,
every time, not just when the estimate happens to be small.

## Running a decision-focused readout

```text
Structure that keeps a readout from becoming a methods lecture:
1. The decision this informs (one sentence)
2. The answer (one sentence, in business terms)
3. Confidence and key caveat (one or two sentences)
4. Recommended next action
5. [Appendix, not the main body] — methodology, robustness checks,
   alternative specifications, for whoever wants to go deeper
```

Leading with methodology is the most common way a technically excellent
analysis loses its audience in the first three minutes — the structure
above front-loads exactly what a time-pressed stakeholder needs to act,
and preserves full rigor for anyone (including a more technical
stakeholder) who wants to verify it in the appendix.

## Cheat sheet

| Situation | Move |
|---|---|
| Vague request arrives | Ask what decision it feeds before starting the analysis |
| Result contradicts stakeholder's intuition | Distinguish "wrong about data" vs. "disagree on action" |
| Result is statistically uncertain | State the interval and its decision implication explicitly |
| Readout to a time-pressed exec | Decision → answer → confidence → action, methods in appendix |
| Feeling pressure to soften a result | Name the instinct; present the finding transparently anyway |

## How It Actually Works

**"The 90% CI includes zero" is a precise statistical fact worth unpacking
for exactly this scenario.** A confidence interval that spans zero means
the observed data is statistically consistent with "no true effect" at the
stated confidence level — equivalently, a formal hypothesis test of "the
lift is zero" against this same data would *fail* to reject the null at
`α = 0.10`. Reporting the 3.4% point estimate alone presents one sample
draw from a distribution of plausible outcomes as if it were a settled
fact; reporting the interval is reporting the actual shape of what the
data supports, which for a stakeholder deciding whether to commit budget
is the entire decision-relevant content of the analysis — the point
estimate is the single most likely value, not the range of values
consistent with the evidence.

**The clarifying-question step formally reduces the value-of-information
mismatch between the analysis performed and the decision it needs to
inform.** A broad root-cause analysis answering "why did revenue drop"
carries information relevant to many possible follow-up decisions;
a narrow decision ("pause the Q2 pricing change, yes or no") only needs the
subset of that information bearing directly on the pricing change. Time
spent characterizing causes irrelevant to that specific decision is not
wasted in an absolute sense, but it is wasted *relative to the decision
deadline* — which is exactly the failure mode "good analysis, wrong
question" describes: the analysis was correct and thorough, but its
information content didn't overlap enough with what the decision actually
needed, discovered only after the effort was already spent.

**Distinguishing "wrong about the data" from "disagree about the action"
maps onto the actual logical structure of the disagreement**: the first is
a claim about whether the estimate itself is correct (checkable — rerun
the numbers, look for a bug, a confounder, a data quality issue) and has a
factual resolution; the second is a claim about what to *do* given an
estimate both sides now agree is correct, which depends on risk tolerance,
competing priorities, and values the data alone can't settle. Treating a
disagreement about action as if it were a disagreement about data (or vice
versa) means arguing past each other indefinitely, because the evidence
that would resolve one type of disagreement is irrelevant to the other.

## 🔀 Related lessons on other tracks

- [Product Lead — Advanced Stakeholder Management (Board-Level)](https://sigilipelli.github.io/product-lead-mastery-path/level-3/09-board-level-stakeholder-management/)
- [Project Manager — 07 · Stakeholder Management Basics](https://sigilipelli.github.io/project-manager-mastery-path/level-1/07-stakeholder-management-basics/)
- [AI Manager — 05 · Managing Cross-Functional AI Teams](https://sigilipelli.github.io/ai-manager-mastery-path/level-2/05-managing-cross-functional-ai-teams/)

## Exercise

Take an analysis you've done in an earlier module. Write the five-part
decision-focused readout structure above for it in under 150 words total
(everything else goes in an "appendix" bullet list). Then write the single
clarifying question you should have asked before starting, if you didn't
already know the real decision behind the request.
