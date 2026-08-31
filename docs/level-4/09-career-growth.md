# 09 · Career Growth: IC to Principal Data Scientist

The skills that get someone promoted from junior to senior (technical
correctness, independence) are not the same skills that get someone from
senior to staff/principal (scope, leverage, organizational judgment).
This module covers what actually changes at each level and how to build
evidence of it deliberately, rather than waiting for it to be recognized.

## What actually changes across levels

```text
Junior:    Executes well-scoped tasks with guidance. Correctness is the bar.
Mid:       Owns a project end-to-end. Scopes its own smaller tasks.
Senior:    Owns ambiguous problems. Decides WHAT to work on, not just how.
           Technical judgment trusted without close review.
Staff:     Influences direction across multiple teams without direct
           authority over them. Work often looks like: identifying a
           problem no one assigned, building consensus, and having
           OTHER people execute much of it.
Principal: Sets technical/strategic direction at an org level. Judged on
           org-wide outcomes, not on projects with your name on them.
```

The senior→staff transition is where most data scientists get stuck, and
it's usually not a skills gap — it's that staff-level work often produces
less individually-attributable output (you convinced three teams to adopt
a shared metric definition; you didn't ship a model) which is harder to
put in a promotion packet than "I shipped model X."

## Building evidence of scope, not just skill

```text
Weak promotion evidence: "I'm very good at causal inference."
                          (a skill claim — the bar for this is already
                          "correct," it doesn't differentiate levels)

Strong promotion evidence: "I noticed three teams were computing 'active
                            user' differently, which was causing
                            conflicting numbers in exec reporting. I
                            proposed and got buy-in on a single
                            definition, now used org-wide, which
                            resolved a recurring source of exec distrust
                            in our metrics."
```

The strong version demonstrates the staff-level pattern directly:
identifying a problem that wasn't assigned, working across team
boundaries without formal authority, and producing an outcome measured
at the org level rather than the project level. Deliberately seeking out
(or creating) opportunities that fit this pattern — rather than only
executing assigned projects extremely well — is the actual lever for
this transition.

## A self-audit framework

```python
import pandas as pd

self_audit = pd.DataFrame({
    "dimension": ["Technical depth", "Scope of ownership", "Cross-team influence",
                  "Mentorship/multiplier effect", "Business impact framing"],
    "current_evidence": [
        "Strong — shipped 3 production models this year",
        "Medium — owns one team's roadmap, not cross-team",
        "Weak — mostly works within own team",
        "Weak — no formal mentees, doesn't review others' work",
        "Medium — reports metrics, rarely ties to $ decisions",
    ],
    "target_for_next_level": [
        "Maintain",
        "Own a cross-team initiative",
        "Lead one initiative requiring buy-in from 2+ other teams",
        "Take on 1-2 mentees; review others' analyses regularly",
        "Reframe next quarter's work explicitly in decision/dollar terms",
    ],
})
print(self_audit.to_string(index=False))
```

Running this kind of audit honestly — and specifically identifying the
one or two dimensions with the weakest current evidence — turns "get
promoted" from a vague goal into concrete, plannable actions for the next
one or two quarters, rather than something that happens passively as a
side effect of doing good project work.

## The generalist-vs-specialist question

```text
Generalist path (most common at senior+):
  Broad competence across the DS stack (analysis, modeling, some
  engineering, stakeholder work). Valuable for ambiguous, cross-cutting
  problems. Ceiling: staff/principal generalist or people management.

Specialist/IC path (viable, less common):
  Deep expertise in one area (causal inference, NLP, a specific domain).
  Valuable when the org has enough volume of that specific problem type
  to justify a dedicated expert. Ceiling: distinguished/principal
  specialist, often org-wide technical authority in that narrow area.
```

Neither path is strictly "more senior" — the mistake is drifting into
specialization by default (because it's comfortable) without checking
whether the org actually has enough volume of that specific problem to
sustain a specialist career track, versus deliberately choosing it because
the organization's needs and your interests both point that way.

## Internal vs. external growth

```text
Signals it might be time to look externally rather than push for
internal promotion:
- The org genuinely lacks staff+ scope of problems (small company,
  thin DS function) — no amount of individual growth creates scope
  that doesn't exist
- Consistently told "not yet" without a specific, actionable gap named
- Your growth area (e.g. ML infra) isn't a direction the org is investing in

Signals internal growth is still the better bet:
- Concrete gaps were named and are addressable within 1-2 quarters
- The org is growing / has emerging scope you could plausibly claim first
- You have unusually good institutional context that resets to zero elsewhere
```

## Cheat sheet

| Level | Differentiator |
|---|---|
| Junior → Mid | Independence on well-scoped work |
| Mid → Senior | Owns ambiguous problems, decides what to work on |
| Senior → Staff | Cross-team influence without formal authority; org-level outcomes |
| Staff → Principal | Sets direction across the org, judged on org-wide outcomes |
| Any level | Multiplier effect (mentorship, reusable frameworks) compounds faster than solo output |

## Exercise

Fill out the self-audit table above honestly for your actual current
situation (or a plausible hypothetical). Pick the single weakest
dimension and write a concrete, one-quarter action plan to build evidence
on it — not a skill to learn, but a specific initiative or opportunity you
could pursue or create.
