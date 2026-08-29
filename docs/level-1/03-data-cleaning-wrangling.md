# 03 · Data Cleaning & Wrangling

Real data is never clean. Duplicate rows, inconsistent text casing, mixed
date formats, and missing values show up in almost every dataset you'll ever
load. This module builds a repeatable checklist for turning a raw export
into something safe to analyze.

## A realistically messy dataset

```python
import numpy as np
import pandas as pd

raw = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 104, 106],
    "signup_date": ["2024-01-15", "2024/02/03", "2024-02-20", None, None, "2024-03-01"],
    "plan": [" Pro", "basic", "PRO", "Basic", "Basic", "enterprise "],
    "monthly_spend": [49.99, 9.99, 49.99, np.nan, np.nan, 199.0],
})
print(raw)
```

```text
   customer_id signup_date         plan  monthly_spend
0          101  2024-01-15          Pro          49.99
1          102  2024/02/03        basic           9.99
2          103  2024-02-20          PRO          49.99
3          104         NaN        Basic            NaN
4          104         NaN        Basic            NaN
5          106  2024-03-01  enterprise          199.00
```

This one small table already has four separate problems: duplicate rows
(customer 104 twice), inconsistent date formats, inconsistent text
capitalization/whitespace in `plan`, and missing values.

## Step 1 — find what's wrong before fixing anything

```python
print(raw.isna().sum())
print(raw.duplicated(subset="customer_id"))
```

```text
customer_id      0
signup_date      2
plan             0
monthly_spend    2
dtype: int64
0    False
1    False
2    False
3    False
4     True
5    False
dtype: bool
```

Always run `.isna().sum()` and check for duplicates *before* touching
anything — deciding how to handle missing/duplicate data is a judgment call,
and you need to know the scope of the problem first.

## Step 2 — remove duplicates

```python
df = raw.drop_duplicates(subset="customer_id", keep="first").copy()
print(df)
```

```text
   customer_id signup_date         plan  monthly_spend
0          101  2024-01-15          Pro          49.99
1          102  2024/02/03        basic           9.99
2          103  2024-02-20          PRO          49.99
3          104         NaN        Basic            NaN
5          106  2024-03-01  enterprise          199.00
```

`keep="first"` keeps the first occurrence of each `customer_id`. The `.copy()`
avoids a `SettingWithCopyWarning` later when you start assigning new columns
onto a filtered DataFrame.

## Step 3 — normalize inconsistent text

```python
df["plan"] = df["plan"].str.strip().str.lower()
print(df["plan"].unique())
```

```text
<StringArray>
['pro', 'basic', 'enterprise']
Length: 3, dtype: str
```

Before this, `" Pro"`, `"PRO"`, and `"pro"` would have been treated as three
different categories by anything downstream (a `groupby`, a plot legend, a
model). `.str.strip().str.lower()` is a two-line fix that prevents a whole
class of silent bugs.

## Step 4 — parse inconsistent dates

```python
df["signup_date"] = pd.to_datetime(df["signup_date"], format="mixed")
print(df["signup_date"])
```

```text
0   2024-01-15
1   2024-02-03
2   2024-02-20
3          NaT
5   2024-03-01
Name: signup_date, dtype: datetime64[us]
```

`format="mixed"` lets pandas infer the format per-value, which handles the
`"2024-01-15"` vs. `"2024/02/03"` inconsistency in one call. A value that
can't be parsed at all becomes `NaT` ("Not a Time") rather than crashing the
whole conversion.

## Step 5 — handle missing values deliberately

```python
median_spend = df["monthly_spend"].median()
print(median_spend)          # 49.99

df["monthly_spend"] = df["monthly_spend"].fillna(median_spend)
print(df)
```

```text
49.99
   customer_id signup_date        plan  monthly_spend
0          101  2024-01-15         pro          49.99
1          102  2024-02-03       basic           9.99
2          103  2024-02-20         pro          49.99
3          104         NaT       basic          49.99
5          106  2024-03-01  enterprise         199.00
```

There is no universally "correct" way to fill missing data — the median is a
reasonable, outlier-resistant default for a skewed numeric column like
spend, but always document *what* you filled and *why* in your analysis.
The alternatives worth knowing:

| Strategy | When to use it | Risk |
|---|---|---|
| `dropna()` | Missingness is rare and random | Loses real data; can bias results if missingness isn't random |
| `fillna(mean/median)` | Numeric column, need a complete dataset for modeling | Understates real variance |
| `fillna(mode)` | Categorical column | Can overrepresent the majority category |
| `fillna(method="ffill")` | Time series, value "carries forward" | Wrong if the true value actually changed |
| Leave as `NaN`/`NaT` | Aggregations (`.mean()`, `.sum()`) that already skip missing values by default | None — often the safest choice |

## Step 6 — verify the final dtypes

```python
print(df.dtypes)
```

```text
customer_id               int64
signup_date      datetime64[us]
plan                        str
monthly_spend           float64
dtype: object
```

A cheap, high-value habit: check `.dtypes` after every cleaning pass. A date
column that's still a string, or a numeric column that got coerced to text
by one stray non-numeric value, causes bugs much further downstream that are
harder to trace back to their source.

## Cheat sheet

| Task | Code |
|---|---|
| Count missing per column | `df.isna().sum()` |
| Find duplicate rows | `df.duplicated(subset=["col"])` |
| Drop duplicates | `df.drop_duplicates(subset=["col"], keep="first")` |
| Normalize text | `df["c"].str.strip().str.lower()` |
| Parse mixed-format dates | `pd.to_datetime(df["c"], format="mixed")` |
| Fill missing numeric | `df["c"].fillna(df["c"].median())` |
| Rename columns | `df.rename(columns={"old": "new"})` |
| Check types after cleaning | `df.dtypes` |

## Exercise

Take the cleaned `df` above and add one more issue to fix: a stray outlier
row with `monthly_spend = 999999` (a clear data-entry error, not a real
enterprise customer). Detect it using the IQR rule (`Q1 - 1.5*IQR` to
`Q3 + 1.5*IQR`, covered fully in Module 04), decide whether to drop it or cap
it, and justify your choice in one sentence.
