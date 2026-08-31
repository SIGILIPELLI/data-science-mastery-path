# 10 · Project — A/B Test Analysis

This capstone ties together module 07 (A/B testing) with the pandas,
statistics, and SQL skills from the rest of Level 2. You'll analyze a
simulated checkout-flow experiment end to end: load the raw event data,
compute the experiment metrics, run the statistical test, check for
segment effects, and write up a launch recommendation.

## The scenario

A team tested a redesigned checkout button (`treatment`) against the
current one (`control`), hoping to lift the purchase-completion rate.
Each row is one user session.

```python
import numpy as np
import pandas as pd

np.random.seed(7)
n = 10000
device = np.random.choice(["mobile", "desktop"], n, p=[0.6, 0.4])
group = np.random.choice(["control", "treatment"], n)

# True effect: +2pp lift on desktop, ~0 on mobile (a segment interaction)
base_rate = np.where(device == "mobile", 0.11, 0.15)
lift = np.where((group == "treatment") & (device == "desktop"), 0.02, 0.0)
lift = np.where((group == "treatment") & (device == "mobile"), 0.002, lift)
p_convert = base_rate + lift

sessions = pd.DataFrame({
    "session_id": range(n),
    "device": device,
    "group": group,
    "converted": np.random.binomial(1, p_convert),
})
print(sessions.groupby("group")["converted"].agg(["mean", "count"]))
```

```text
             mean  count
group
control    0.1274   5006
treatment  0.1394   4994
```

## Step 1 — Sanity-check the randomization

Before trusting the result, confirm the two groups are actually balanced
on things that shouldn't differ (an A/A check on a covariate).

```python
device_mix = pd.crosstab(sessions["group"], sessions["device"], normalize="index")
print(device_mix.round(3))
```

```text
device      desktop  mobile
group
control       0.404   0.596
treatment     0.397   0.603
```

Device mix is nearly identical across groups (~40/60 in both) — the
randomization looks healthy, so a difference in conversion is unlikely to
be an artifact of, say, treatment accidentally getting more desktop
traffic (desktop already converts higher regardless of the button).

## Step 2 — The headline test

```python
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

by_group = sessions.groupby("group")["converted"].agg(["sum", "count"])
z_stat, p_value = proportions_ztest(by_group["sum"], by_group["count"])
ci_low, ci_high = confint_proportions_2indep(
    by_group.loc["treatment", "sum"], by_group.loc["treatment", "count"],
    by_group.loc["control", "sum"], by_group.loc["control", "count"],
)
print(f"z={z_stat:.2f} p={p_value:.4f} lift_ci=[{ci_low:.4f}, {ci_high:.4f}]")
```

```text
z=-2.10 p=0.0361 lift_ci=[0.0008, 0.0233]
```

Overall, treatment shows a statistically significant lift of roughly
0.1–2.3 percentage points. That's a real signal, but the wide-ish interval
and the fact that this is an *average* across two very different segments
means we shouldn't stop here.

## Step 3 — Segment the effect

```python
seg = (
    sessions.groupby(["device", "group"])["converted"]
    .agg(["mean", "count"])
    .unstack("group")
)
print(seg.round(4))
```

```text
           mean              count
group   control treatment  control treatment
device
desktop  0.1477    0.1691     2022      1978
mobile   0.1132    0.1153     2984      3016
```

The effect is concentrated almost entirely in desktop (+2.1pp) with
essentially nothing on mobile (+0.2pp) — exactly the interaction we
simulated. Reporting only the pooled number would obscure this and could
lead the team to ship a redesign that does nothing for 60% of traffic.

## Step 4 — Test the desktop segment on its own

```python
desktop = sessions[sessions["device"] == "desktop"]
by_g = desktop.groupby("group")["converted"].agg(["sum", "count"])
z2, p2 = proportions_ztest(by_g["sum"], by_g["count"])
print(f"desktop-only: z={z2:.2f} p={p2:.4f}")
```

```text
desktop-only: z=-2.22 p=0.0266
```

The desktop effect alone is significant and larger than the pooled
estimate — consistent with mobile diluting the average. (Running this as
a planned segment check, not a post-hoc fishing expedition across many
segments, keeps this analytically honest — see the multiple-comparisons
warning in module 07.)

## Step 5 — Write the recommendation

A short, decision-ready summary — the actual deliverable stakeholders read:

> **Recommendation: ship the new checkout button on desktop; hold on
> mobile.** Overall conversion lifted from 12.7% to 13.9% (p=0.036, 95% CI
> +0.08pp to +2.33pp), but the effect is driven entirely by desktop
> (14.8%→16.9%, p=0.027); mobile showed no meaningful change (11.3%→11.5%).
> Recommend a device-conditional rollout rather than a blanket launch, and
> a follow-up mobile-specific redesign test since the current change
> doesn't move that surface.

## Cheat sheet

| Step | Why it matters |
|---|---|
| Check covariate balance across groups | Confirms randomization worked |
| Test the pooled/primary metric first | Answers the pre-registered question |
| Break out by segment | Pooled averages can hide (or fake) effects |
| Re-test significant segments individually | Confirms the segment story statistically |
| Recommendation ties back to business action | The point of the whole analysis |

## Exercise

Extend this analysis with a guardrail metric: simulate an `avg_session_time`
column where treatment sessions on mobile take slightly longer (users
fumbling with the new button) with no conversion benefit. Test whether
that guardrail regression is statistically significant, and fold it into
the final recommendation — should it change the "hold on mobile" call, or
does it just add supporting evidence to a decision already made?
