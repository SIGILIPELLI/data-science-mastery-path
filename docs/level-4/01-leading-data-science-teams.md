# 01 · Leading Data Science Teams

Moving from senior IC to team lead changes the job from "produce good
analysis" to "create the conditions for a team to produce good analysis
reliably." This module covers the concrete mechanics: how to structure a
small DS team, run useful 1:1s and reviews, and avoid the most common
failure modes of new managers.

## Team topologies at different sizes

```text
1-2 DS embedded in a product team
  → generalists; no dedicated lead; report to a PM or eng manager

3-6 DS, one team
  → first DS manager role; still hands-on; mix of generalist + 1 specialist

6-15 DS, multiple pods
  → manager of managers or manager + senior ICs as tech leads per pod
  → need shared standards (below) or every pod reinvents tooling

15+ DS across the org
  → platform/tooling team emerges (see Module 03) to serve pods
  → risk of duplicated effort without a central function or guild
```

The inflection point most new leads miss is 3→6: below that, informal
coordination (a weekly sync) works; above it, someone needs to own
project prioritization explicitly or the team's time gets consumed by
whoever asks loudest.

## Running a useful 1:1

```text
Bad 1:1 (status report):
  "What did you work on this week? ... OK, next."

Better 1:1 (skip-level-worthy):
  1. Career/growth check-in (not every week, but on a visible cadence)
  2. Blockers only THEY can't resolve (if they can resolve it, coach, don't solve)
  3. One question you actually don't know the answer to
  4. Explicit "anything I should know that isn't in Slack/Jira"
```

A status report duplicates what a standup or project tracker already
covers; a 1:1's marginal value is the things that don't fit in either —
career trajectory, interpersonal friction, and half-formed concerns the
report hasn't fully articulated yet. Protecting that time from turning
into status-reporting is a manager's job, not the report's.

## Reviewing analysis without doing it yourself

The hardest transition for a former IC is reviewing work you'd have done
differently without either rewriting it or rubber-stamping it.

```text
Review checklist (ask, don't dictate):
- Does the analysis answer the question that was actually asked?
- What would change your conclusion? Did you check that?
- What's the weakest assumption, and how sensitive is the result to it?
- Would this survive a skeptical stakeholder's first three questions?
- Is the level of rigor matched to the decision's stakes? (a one-off
  Slack answer doesn't need the same bar as a pricing model change)
```

Asking these as questions in a review — rather than rewriting the notebook
— does two things a rewrite doesn't: it surfaces whether the analyst
understood their own reasoning (vs. pattern-matched a template), and it
scales, since a lead who rewrites every review becomes the bottleneck the
team is structured to avoid.

## Prioritization: a lightweight scoring frame

```python
import pandas as pd

projects = pd.DataFrame({
    "project": ["Churn model v2", "Ad-hoc exec request", "Data quality fixes", "New dashboard"],
    "impact": [8, 3, 6, 4],       # 1-10, expected business value if it works
    "confidence": [6, 9, 9, 8],   # 1-10, how sure are we impact estimate is right
    "effort_weeks": [6, 0.5, 2, 3],
})
projects["score"] = (projects["impact"] * projects["confidence"] / 10) / projects["effort_weeks"]
print(projects.sort_values("score", ascending=False))
```

```text
             project  impact  confidence  effort_weeks  score
1 Ad-hoc exec request       3           9           0.5  5.400
2 Data quality fixes        6           9           2.0  2.700
3      New dashboard        4           8           3.0  1.067
0    Churn model v2         8           6           6.0  0.800
```

A simple impact × confidence / effort score (a lightweight RICE-style
frame) makes prioritization conversations concrete instead of political —
it doesn't replace judgment (a low-scoring strategic bet may still be
worth doing), but it forces the "why" for deviating from the score to be
stated explicitly rather than left implicit.

## Common new-manager failure modes

| Failure mode | What it looks like | Fix |
|---|---|---|
| Still the best IC on the team | You take the hardest project yourself every time | Delegate the hard project; coach through it |
| Conflict avoidance | Underperformance goes unaddressed for months | Address it directly and early, in private |
| No visible prioritization | Team works on whatever was asked most recently | Use a scoring frame; say no explicitly and in writing |
| 1:1s become status reports | No career or blocker conversation happens | Bring your own agenda items, not just theirs |
| Over-indexing on one skip-level's ask | Team roadmap driven by whoever escalates loudest | Route requests through a single intake/prioritization step |

## Cheat sheet

| Tool | Use it for |
|---|---|
| Team topology by size | Deciding when to split into pods / add tech leads |
| 1:1 agenda template | Keeping 1:1s from degrading into status updates |
| Review-by-questions | Scaling review without becoming the bottleneck |
| Impact × confidence / effort | Making prioritization tradeoffs explicit and defensible |
| Failure-mode table | Self-checking your own first six months as a lead |

## Exercise

For your current (or a hypothetical) team's project backlog, score at
least five projects with the impact/confidence/effort frame above. Write
one paragraph justifying one deliberate deviation from the resulting
ranking — a case where the score says one thing but you'd still choose
differently, and why.
