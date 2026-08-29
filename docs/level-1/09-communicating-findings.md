# 09 · Communicating Findings

An analysis nobody understands or acts on has zero impact, no matter how
correct the statistics behind it are. This module covers visual storytelling
— structuring a finding so a non-technical reader gets it in seconds — and
the misleading-chart tricks to actively avoid, some of which you've already
seen hinted at in earlier modules.

## The revenue example

```python
import pandas as pd
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
revenue = [102000, 104500, 103800, 106200, 105900, 108100]
df = pd.DataFrame({"month": months, "revenue": revenue})
print(df)

pct_change = (revenue[-1] - revenue[0]) / revenue[0] * 100
print(round(pct_change, 2))
```

```text
  month  revenue
0   Jan   102000
1   Feb   104500
2   Mar   103800
3   Apr   106200
4   May   105900
5   Jun   108100
5.98
```

Revenue grew a real but modest 5.98% over six months. Here's the same data
plotted two ways:

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(months, revenue, marker="o")
axes[0].set_ylim(0, 120000)
axes[0].set_title("Honest: y-axis from 0")

axes[1].plot(months, revenue, marker="o")
axes[1].set_ylim(100000, 109000)
axes[1].set_title("Misleading: truncated y-axis")

fig.savefig("revenue_comparison.png", dpi=150)
```

The left chart (y-axis 0–120,000) shows a gentle, honest upward slope — it
visually matches the real 5.98% growth. The right chart (y-axis
100,000–109,000) shows the *identical* six numbers as a dramatic,
near-vertical climb, because zooming the axis into a tiny window
exaggerates every wiggle. Both charts are technically "accurate" — same
data, same axis labels — but only the left one gives an honest visual
impression of the magnitude of change. This is the exact trap flagged as a
warning in Module 05's bar chart section, and it applies to line charts too.

!!! danger "Rule of thumb"
    For charts meant to convey *how big* a change or difference is, start
    the axis at zero. Zooming in is only defensible when the chart's whole
    purpose is to show fine-grained wiggle in an already-established trend
    (e.g. a stock trader's minute-by-minute chart) — and even then, label
    the axis range prominently so the reader isn't misled.

## Visual storytelling: structure over decoration

A good data science write-up follows a predictable shape, in this order:

1. **The headline finding, first.** "Revenue grew 6% in H1, driven almost
   entirely by the West region" — not "here are twelve charts, you figure
   it out."
2. **One chart per finding**, not one mega-dashboard. If a chart needs a
   paragraph of caption to explain what it's showing, it's usually two
   charts trying to be one.
3. **Comparison, not just description.** "$108,100 in June" means little
   alone; "$108,100 in June, up from $102,000 in January, in line with our
   Q2 target" is a sentence someone can act on.
4. **Uncertainty stated plainly**, using the vocabulary from Module 06 —
   "revenue is up 6%, and this is a real trend, not month-to-month noise
   (p < 0.01)" is a stronger and more honest claim than a bare percentage.
5. **A recommendation or "so what"**, echoing Module 01's "so what?" test —
   tie every finding back to a decision.

## A checklist of chart tricks to avoid

| Trick | Why it misleads | Fix |
|---|---|---|
| Truncated y-axis on a bar/column chart | Exaggerates small differences | Start bar-chart axes at 0 |
| Dual y-axes with different scales | Makes unrelated lines look correlated | Use two separate charts, or one shared axis |
| Cherry-picked date range | Hides that a "trend" is really noise or a one-off event | Show a longer baseline period alongside |
| Pie chart with many thin slices | Humans compare angles poorly | Use a sorted bar chart instead |
| Reporting a % change with no baseline number | "grew by 200%!" from 1 to 3 units sounds huge | Always show the absolute numbers alongside the % |
| Correlation implied as causation in a caption | Overclaims what the data shows | State explicitly: "associated with," not "caused by" (Module 07) |
| Showing only the statistically significant result | Classic p-hacking (Module 06) presented visually | Show all comparisons tested, not just the "winner" |

## Cheat sheet

| Task | Guidance |
|---|---|
| Bar/column chart axis | Always start at 0 |
| Line chart showing magnitude of change | Start at 0 unless zooming is the explicit point, and label clearly |
| Leading with the finding | State it in one sentence before showing any chart |
| Reporting a %, always also report | The absolute numbers behind it |
| Claiming a relationship | Say "associated with," never "causes," unless from a controlled experiment |
| One chart, one point | Split a chart that needs a paragraph to explain |

## Exercise

Take any bar chart you've made in this track so far (Module 05's
`spend_by_region` chart is a good candidate) and write the three-sentence
summary a non-technical stakeholder should read *before* seeing the chart:
(1) the headline finding, (2) the comparison that gives it context, (3) one
honest caveat about what the data does or doesn't prove.
