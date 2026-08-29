# 04 · Exploratory Data Analysis (EDA)

EDA is the deliberate, structured process of looking at a dataset before you
try to answer any specific question with it — checking its shape, its
distributions, its outliers, and its relationships between columns. Skipping
this step is how analyses get built on data that's silently broken.

## A synthetic customer dataset

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n = 200
df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "income": np.concatenate([rng.normal(45000, 12000, n - 3), [250000, 260000, 5000]]),
    "tenure_months": rng.integers(1, 60, n),
    "churned": rng.choice([0, 1], size=n, p=[0.75, 0.25]),
})
print(df.shape)
```

```text
(200, 4)
```

Three deliberately-planted problem rows are hiding in `income` — this
mirrors how real income/revenue columns almost always contain a handful of
extreme values.

## Step 1 — shape and summary statistics

```python
print(df.describe())
```

```text
             age         income  tenure_months     churned
count  200.00000     200.000000     200.000000  200.000000
mean    43.96500   47289.756071      28.380000    0.280000
std     16.05749   23775.380090      17.237496    0.450126
min     18.00000    5000.000000       1.000000    0.000000
25%     30.00000   38062.068815      14.000000    0.000000
50%     43.50000   46314.203943      26.500000    0.000000
75%     58.25000   52238.692691      43.250000    1.000000
max     69.00000  260000.000000      59.000000    1.000000
```

The first thing to check in any `.describe()` output: does `max` look
wildly larger than `75%`? Here income's `max` (260,000) is nearly 5x the
75th percentile (52,239) — a strong signal of outliers worth investigating
before you compute anything like a mean that outliers would distort.

## Step 2 — find outliers with the IQR rule

The interquartile range (IQR) rule flags any value more than 1.5×IQR beyond
the 25th/75th percentiles as a likely outlier:

```python
q1 = df["income"].quantile(0.25)
q3 = df["income"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

outliers = df[(df["income"] < lower) | (df["income"] > upper)]
print(lower, upper)
print(outliers)
```

```text
16797.133000439164 73503.62850511285
     age         income  tenure_months  churned
18    68   75834.138156             44        0
197   25  250000.000000             34        0
198   46  260000.000000             21        0
199   50    5000.000000             29        0
```

All three planted problem rows were caught, plus one naturally-occurring
high earner. What to do next is a judgment call: cap them (winsorize),
investigate whether they're data-entry errors, or leave them if they're
real, meaningful data points — but you must *know they're there* before
deciding, and now you do.

## Step 3 — look at categorical/binary columns with value_counts

```python
print(df["churned"].value_counts())
print(df["churned"].value_counts(normalize=True).round(3))
```

```text
churned
0    144
1     56
Name: count, dtype: int64
churned
0    0.72
1    0.28
Name: proportion, dtype: float64
```

`normalize=True` converts counts to proportions — "28% of customers
churned" is usually a more useful sentence than "56 customers churned"
because it's comparable across datasets of different sizes.

## Step 4 — compare groups

```python
print(df.groupby("churned")[["age", "income", "tenure_months"]].mean().round(1))
```

```text
          age   income  tenure_months
churned
0        44.1  47502.3           29.3
1        43.5  46743.3           26.0
```

This is the single most common EDA move: "does this outcome column look
different across groups of the other columns?" Here churned and non-churned
customers look fairly similar on average — a hint (not proof; that requires
the hypothesis testing from Module 06) that these three features alone may
not explain churn well.

## Step 5 — correlations between numeric columns

```python
print(df.corr(numeric_only=True).round(2))
```

```text
                age  income  tenure_months  churned
age            1.00   -0.03           0.00    -0.02
income        -0.03    1.00           0.02    -0.01
tenure_months  0.00    0.02           1.00    -0.09
churned       -0.02   -0.01          -0.09     1.00
```

Read the diagonal (always 1.0 — a column perfectly correlates with itself)
and everything else relative to it. All the off-diagonal values here are
close to 0, meaning none of `age`, `income`, or `tenure_months` moves
strongly with `churned` in this synthetic dataset — a real finding you'd
report as "none of these three features show a strong linear relationship
with churn," not something to paper over.

!!! warning "Correlation is not the whole story"
    `.corr()` only measures **linear** relationships. Two columns can be
    strongly related in a curved or categorical way and still show a
    correlation near 0. Module 07 covers this limitation in depth, alongside
    the "correlation ≠ causation" trap.

## The EDA checklist

Run through this on every new dataset, before writing a single conclusion:

1. `df.shape`, `df.dtypes`, `df.head()` — what am I even looking at?
2. `df.isna().sum()` — where's the missing data? (Module 03)
3. `df.describe()` — do the min/max/percentiles look sane?
4. IQR or a boxplot per numeric column — where are the outliers?
5. `value_counts()` on every categorical/binary column
6. `groupby` comparisons across any column you suspect matters
7. `.corr()` between numeric columns (with the linear-only caveat above)

## Cheat sheet

| Question | Code |
|---|---|
| What am I looking at? | `df.shape`, `df.dtypes`, `df.head()` |
| Summary stats | `df.describe()` |
| Outlier bounds (IQR rule) | `q1 - 1.5*iqr`, `q3 + 1.5*iqr` |
| Category frequencies | `df["c"].value_counts(normalize=True)` |
| Compare groups | `df.groupby("g")[["c1","c2"]].mean()` |
| Linear relationships | `df.corr(numeric_only=True)` |

## Exercise

Using the `df` from this module, run the full EDA checklist on
`tenure_months` alone: report its mean, median, IQR outlier bounds, and
whether it has any outliers. Then answer in one sentence — based on the
`groupby` and `.corr()` output above, would you recommend building a churn
model on just these three features, or would you first want more data?
Justify your answer using the specific correlation numbers.
