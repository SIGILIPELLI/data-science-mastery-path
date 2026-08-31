# 03 · Advanced Visualization (plotly)

Level 1's matplotlib charts are static images — great for reports, limited
for exploration. **Plotly** produces interactive charts (hover tooltips,
zoom, pan, toggle series) that render in Jupyter, HTML, or a browser, with
almost the same amount of code.

## A basic interactive scatter

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score":       [52, 55, 61, 60, 68, 71, 75, 80, 84, 88],
    "cohort":      ["A", "A", "A", "B", "B", "A", "B", "B", "A", "B"],
})

fig = px.scatter(
    df, x="study_hours", y="score", color="cohort",
    trendline="ols", title="Study hours vs. score by cohort",
)
fig.write_html("study_scores.html")
print(fig.data[0].type, len(fig.data))
```

```text
scatter 4
```

`px.scatter` with `color="cohort"` automatically splits the data into
colored series with a legend and hover labels — no manual loop over
groups. `trendline="ols"` fits and overlays a regression line per group
(requires `statsmodels` installed), doubling the trace count from 2 to 4.
`fig.write_html(...)` saves a fully interactive standalone file you can
open in any browser — no server needed.

## Faceted small multiples

When comparing more than 2-3 groups, one chart per group ("faceting") beats
cramming everything into one:

```python
sales = pd.DataFrame({
    "month":  list(range(1, 7)) * 2,
    "region": ["East"] * 6 + ["West"] * 6,
    "revenue": [100, 120, 115, 140, 160, 150, 90, 95, 130, 125, 145, 170],
})

fig = px.line(
    sales, x="month", y="revenue", facet_col="region",
    markers=True, title="Monthly revenue by region",
)
print([t.name for t in fig.data])
```

```text
['East', 'West']
```

`facet_col="region"` draws one subplot per region sharing the same axes,
which makes shapes directly comparable — far easier to read than one chart
with two overlapping lines once you have more than a couple of categories.

## Interactive bar with hover detail

```python
summary = sales.groupby("region", as_index=False)["revenue"].sum()

fig = px.bar(
    summary, x="region", y="revenue", text="revenue",
    color="region", title="Total revenue by region",
)
fig.update_traces(texttemplate="%{text:,}", textposition="outside")
print(summary.to_dict("records"))
```

```text
[{'region': 'East', 'revenue': 785}, {'region': 'West', 'revenue': 755}]
```

`text="revenue"` labels each bar with its value directly, removing the need
for the viewer to eyeball the y-axis. `update_traces` reformats that label
with thousands separators and moves it above the bar.

## Heatmap for a correlation matrix

```python
import numpy as np

rng = np.random.default_rng(3)
metrics = pd.DataFrame(rng.normal(size=(200, 4)), columns=["a", "b", "c", "d"])
metrics["b"] = metrics["a"] * 0.8 + rng.normal(scale=0.3, size=200)  # correlated with a

corr = metrics.corr().round(2)
fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
print(corr.loc["a", "b"])
```

```text
0.94
```

`px.imshow` on a correlation matrix with `text_auto=True` annotates every
cell with its value and `color_continuous_scale="RdBu_r"` maps positive
correlations to red and negative to blue with white at zero (`zmin`/`zmax`
pin the scale so 0 always lands at white, not just the darkest observed
value).

## When to reach for plotly vs. matplotlib

| Use case | Tool |
|---|---|
| Static chart for a printed report / paper | matplotlib |
| Exploring data interactively in a notebook | plotly |
| Dashboard or app the audience will click around in | plotly (pairs with Dash/Streamlit — Level 3) |
| Full control over every pixel for publication | matplotlib |

## Exercise

Using the `metrics` DataFrame above, add a fifth column `"e"` that is
`metrics["c"]` plus noise so it's moderately correlated (~0.5-0.7). Recompute
the correlation heatmap with `px.imshow` and confirm `e` shows up as a
medium-red cell against `c` but near-white against `b` and `d`. Then build a
faceted scatter (`facet_col`) of `a` vs. `b`, `c` vs. `d`, and `c` vs. `e` to
visually confirm the same relationships the heatmap shows.
