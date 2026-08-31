# 04 · Feature Engineering Basics

Raw columns rarely feed a model well. **Feature engineering** — transforming,
combining, and encoding data into inputs a model can actually learn
from — routinely matters more to model quality than the choice of
algorithm.

## Encoding categorical variables

Models need numbers. Two standard encodings, and when to use each:

```python
import pandas as pd

df = pd.DataFrame({
    "city": ["NY", "LA", "NY", "SF", "LA"],
    "plan": ["basic", "pro", "pro", "basic", "enterprise"],
})

# One-hot: for nominal categories with no inherent order
one_hot = pd.get_dummies(df, columns=["city"], prefix="city")
print(one_hot)
```

```text
         plan  city_LA  city_NY  city_SF
0       basic    False     True    False
1         pro     True    False    False
2         pro    False     True    False
3       basic    False    False     True
4  enterprise     True    False    False
```

```python
# Ordinal: for categories with a real order
plan_order = {"basic": 0, "pro": 1, "enterprise": 2}
df["plan_rank"] = df["plan"].map(plan_order)
print(df)
```

```text
  city        plan  plan_rank
0   NY       basic          0
1   LA         pro          1
2   NY         pro          1
3   SF       basic          0
4   LA  enterprise          2
```

One-hot encoding creates a binary column per category with no implied
ordering — correct for `city`, where "NY > LA" would be meaningless. Ordinal
encoding maps categories to numbers that preserve a real ranking — correct
for `plan`, where enterprise genuinely represents "more" than basic. Using
one-hot on `plan` would throw away the ordering information; using ordinal
on `city` would invent a fake one.

## Scaling numeric features

Many algorithms (k-NN, SVMs, gradient descent-based models, PCA) are
sensitive to feature scale — a column measured in thousands will dominate
one measured in single digits purely due to units, not importance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.DataFrame({"income": [42000, 58000, 39000, 120000, 65000],
                      "age":    [25, 34, 22, 51, 38]})

scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print(np.round(scaled, 2))
print(np.round(scaled.mean(axis=0), 2), np.round(scaled.std(axis=0), 2))
```

```text
[[-0.68 -0.85]
 [ 0.05  0.05]
 [-0.79 -1.15]
 [ 2.09  1.83]
 [ 0.32  0.36]]
[0. 0.] [1. 1.]
```

`StandardScaler` rescales each column to mean 0, std 1 — after scaling,
`income` and `age` are on the same footing, so a distance-based model
won't silently over-weight income just because its raw numbers are bigger.
`MinMaxScaler` (rescales to [0, 1]) is the alternative when you want a
bounded range instead, e.g. for neural network inputs.

!!! warning "Fit the scaler on training data only"
    Always call `.fit()` (or `.fit_transform()`) on the training split, then
    `.transform()` (not `.fit_transform()`) on validation/test data using the
    *same* fitted scaler. Fitting on the full dataset lets test-set
    statistics leak into training — a classic source of over-optimistic
    model scores.

## Creating new features from existing ones

The highest-leverage feature engineering usually isn't a library call — it's
domain knowledge encoded as arithmetic:

```python
txns = pd.DataFrame({
    "signup_date": pd.to_datetime(["2024-01-01", "2024-03-15", "2024-06-01"]),
    "last_purchase": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-01"]),
    "total_spent": [450, 120, 30],
    "num_orders": [9, 3, 1],
})

txns["tenure_days"] = (txns["last_purchase"] - txns["signup_date"]).dt.days
txns["avg_order_value"] = (txns["total_spent"] / txns["num_orders"]).round(2)
print(txns[["tenure_days", "avg_order_value"]])
```

```text
   tenure_days  avg_order_value
0          152             50.0
1           78             40.0
2            0             30.0
```

`avg_order_value` (spend per order) is often far more predictive of future
behavior than `total_spent` alone, because it separates "spends a lot per
purchase" from "has just been around longer and accumulated more orders."
`tenure_days` turns two raw dates into a single interpretable number a model
can use directly.

## Binning a continuous variable

```python
ages = pd.Series([19, 24, 31, 45, 52, 67, 15])
bins = pd.cut(ages, bins=[0, 18, 30, 50, 100], labels=["minor", "young_adult", "adult", "senior"])
print(pd.DataFrame({"age": ages, "age_group": bins}))
```

```text
   age    age_group
0   19  young_adult
1   24  young_adult
2   31        adult
3   45        adult
4   52       senior
5   67       senior
6   15        minor
```

Binning trades precision for robustness and interpretability — useful when
a model (or a human reading the model's rules) benefits from coarse
categories, or when the relationship with the target is non-linear across
ranges rather than a smooth trend.

## Cheat sheet

| Task | Code |
|---|---|
| Nominal category → dummy columns | `pd.get_dummies(df, columns=[...])` |
| Ordered category → rank number | `series.map({"low": 0, "high": 1})` |
| Standardize numeric features | `StandardScaler().fit_transform(X)` |
| Continuous → buckets | `pd.cut(series, bins=[...], labels=[...])` |
| Date difference → numeric | `(date_a - date_b).dt.days` |

## Exercise

Using the `txns` DataFrame above, add a feature `orders_per_month` computed
as `num_orders / (tenure_days / 30)`, guarding against division by zero for
the customer with `tenure_days == 0` (treat that case as `num_orders`
itself). Then one-hot encode a new `channel` column you add
(`["organic", "paid", "organic"]`) and combine it with the scaled numeric
columns into a single model-ready DataFrame.
