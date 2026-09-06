# 05 · Data Visualization Basics

A well-chosen chart answers a question in half a second that a table of
numbers would take a minute to explain. This module covers the four chart
types that cover most data science needs — histogram, scatter, bar, and
box plot — using matplotlib and seaborn.

## Setup and sample data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(3)
n = 200
df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "income": rng.normal(45000, 12000, n),
    "region": rng.choice(["North", "South", "East", "West"], n),
    "spend": rng.normal(120, 40, n),
})
df["spend"] = df["spend"] + df["income"] * 0.001  # correlate spend with income a bit
```

## Histogram — the shape of one numeric column

Use a histogram whenever the question is "what does the distribution of this
column look like?" — is it roughly bell-shaped, skewed, bimodal?

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["income"], bins=20)
ax.set_title("Income distribution")
fig.savefig("income_hist.png", dpi=150)

print(df["income"].min(), df["income"].max())
```

```text
17651.68 75600.42
```

Running this produces a bell-shaped histogram centered near $45,000
(matching the `rng.normal(45000, 12000, n)` generator), ranging from about
$17,650 to $75,600 — no long tail or second hump, so a mean/std summary is a
reasonable way to describe this column (unlike the income data in Module 04,
which had planted outliers that would make a histogram visibly skewed).

## Scatter plot — the relationship between two numeric columns

Use a scatter plot to see whether two numeric columns move together, and
what shape that relationship has (linear, curved, no pattern).

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["income"], df["spend"], alpha=0.5)
ax.set_xlabel("income")
ax.set_ylabel("spend")
fig.savefig("income_vs_spend.png", dpi=150)

print(df["income"].corr(df["spend"]).round(3))
```

```text
0.182
```

The cloud of points slopes gently upward — matching the correlation of
0.182 computed the same way Module 04 introduced. `alpha=0.5` (partial
transparency) is worth using by default on scatter plots: with 200+ points,
overlapping markers otherwise hide how dense a region really is.

## Bar chart — comparing a summary statistic across categories

Use a bar chart for "compare this number across groups" — never for raw
individual data points (that's what a scatter or box plot is for).

```python
fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("region")["spend"].mean().plot(kind="bar", ax=ax)
fig.savefig("spend_by_region.png", dpi=150)

print(df.groupby("region")["spend"].mean().round(1))
```

```text
region
East     162.2
North    168.9
South    175.9
West     178.5
Name: spend, dtype: float64
```

The resulting bar chart shows four bars of similar height (162–179), which
correctly conveys "spend is fairly similar across regions, with West
slightly highest" — the bars all being roughly the same height *is* the
finding.

!!! danger "The most common misleading chart: a truncated y-axis"
    If that bar chart's y-axis started at 150 instead of 0, the same data
    would visually scream "West spends dramatically more!" — a real,
    common trick (intentional or not) for making a small difference look
    large. **Bar charts should almost always start their axis at zero.**
    Line charts showing *change over time* are a legitimate exception where
    zooming in is fine, as long as the axis is clearly labeled.

## Box plot — distribution comparison across categories

A box plot shows the median, IQR (the box), and outliers (the dots) for a
numeric column, split by category — it's the "histogram + groupby" combined
into one chart.

```python
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="region", y="income", ax=ax)
fig.savefig("income_by_region_box.png", dpi=150)
```

The box for each region sits at roughly the same height with similar spread
— consistent with `income` being generated independently of `region` in this
synthetic dataset, and a good check against the bar chart above (which only
showed the mean of `spend`, not the full income distribution per region).

## Choosing the right chart

| Question | Chart |
|---|---|
| What's the shape of one numeric column's distribution? | Histogram |
| Do two numeric columns move together? | Scatter plot |
| Compare one summary number across categories | Bar chart |
| Compare a full distribution across categories | Box plot |
| How does a value change over time? | Line chart |
| What fraction does each category make up? | Bar chart (avoid pie charts — humans are bad at comparing angles) |

## Cheat sheet

| Task | Code |
|---|---|
| Histogram | `ax.hist(df["col"], bins=20)` |
| Scatter | `ax.scatter(df["x"], df["y"], alpha=0.5)` |
| Bar of a groupby | `df.groupby("g")["v"].mean().plot(kind="bar")` |
| Box plot by category | `sns.boxplot(data=df, x="cat", y="num")` |
| Always label axes | `ax.set_xlabel(...)`, `ax.set_ylabel(...)` |
| Save a figure | `fig.savefig("name.png", dpi=150)` |

## How It Actually Works

**How a histogram actually turns numbers into bar heights.** `bins=20`
divides the observed range (`min` to `max`) into 20 equal-width intervals,
then counts how many values fall into each interval — the bar height *is*
that count (or, if `density=True`, the count rescaled so the total area
under all bars sums to 1, which makes histograms of different sample sizes
visually comparable). Bin width matters more than it looks: too few bins
smooth away real structure like a second hump (bimodality); too many bins
turn sampling noise into what looks like meaningful jaggedness. There's no
universally correct bin count — but a common rule of thumb (Sturges' rule,
`k ≈ log₂(n) + 1`) gives a reasonable starting point pandas/matplotlib
approximate by default.

**Why truncating a y-axis distorts perception, mechanically.** A chart
communicates magnitude through the *visual length* of a bar or the *slope*
of a line, both computed as `(pixel_height_of_max - pixel_height_of_min)`
mapped linearly onto the axis range you set. If four bars have values
162–179 mapped onto a 0–200 axis, their pixel-height differences are small
relative to the full bar height — correctly signaling "these are similar."
Map the same four values onto a 150–180 axis instead, and the *identical*
value differences now consume the *entire* available pixel range, so the
rendering engine stretches small numeric gaps into large visual ones. The
data and the axis labels can both be 100% accurate while the visual
impression is completely wrong — the distortion lives entirely in the
pixel-to-value mapping, which is exactly why "start bar charts at zero" is
a rule about the *rendering*, not the data.

**What a box plot's whiskers and box actually encode.** The box spans Q1 to
Q3 (the same IQR from Module 04); the line inside is the median, not the
mean, so it's robust to the same outliers a mean would be pulled by; the
whiskers extend to the most extreme data point still within 1.5×IQR of the
box (the identical Tukey rule from Module 04); anything beyond that is
drawn as an individual dot. A box plot is, mechanically, the IQR outlier
rule and the five-number summary rendered as a picture.

## Exercise

Using the `df` from this module, make a scatter plot of `age` vs. `spend`
and compute their correlation. Then make a bar chart of average `age` per
`region`. Write one sentence for each chart stating what it shows — and for
the bar chart specifically, confirm your y-axis starts at 0.
