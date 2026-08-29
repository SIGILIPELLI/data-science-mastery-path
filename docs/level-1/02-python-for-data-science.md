# 02 · Python for Data Science

Data scientists spend most of their coding time in two libraries: **NumPy**,
which stores numbers in fast, fixed-type arrays, and **pandas**, which wraps
NumPy in labeled, spreadsheet-like tables. This module covers the subset of
each that shows up in nearly every analysis.

## NumPy arrays

```python
import numpy as np

a = np.array([2.0, 4.0, 6.0, 8.0])
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape, a.dtype)   # (4,) float64
print(M.shape)            # (2, 3) -> 2 rows, 3 columns
```

```text
(4,) float64
(2, 3)
```

`shape` is the property you'll check most often — a dataset of 500 rows and
6 columns is `(500, 6)`, and a mismatched shape is the single most common
bug in data code.

Reproducible random numbers (important for anything involving sampling or
simulation later in this track):

```python
rng = np.random.default_rng(seed=42)
print(rng.normal(size=(2, 2)))
```

```text
[[ 0.30471708 -1.03998411]
 [ 0.7504512   0.94056472]]
```

Passing the same `seed` always reproduces the same numbers — this is how you
make a "random" analysis reproducible for a colleague or your future self.

## Vectorized operations

Arithmetic on arrays applies element-wise with no Python loop:

```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0, 30.0])

print(x + y)     # [11. 22. 33.]
print(x * y)     # [10. 40. 90.]
print(x ** 2)    # [1. 4. 9.]
```

```text
[11. 22. 33.]
[10. 40. 90.]
[1. 4. 9.]
```

This is **vectorization**: the loop runs in optimized C rather than
interpreted Python. The practical rule for this whole track: *if you're
writing a `for` loop over rows of data, look for the array/DataFrame
operation that does the same thing instead* — it will be both shorter and
often 10-100x faster.

## Boolean masks — how you'll filter data constantly

```python
v = np.array([15, 22, 31, 40, 55])
mask = v > 25
print(mask)      # [False False  True  True  True]
print(v[mask])   # [31 40 55]
```

```text
[False False  True  True  True]
[31 40 55]
```

Every "show me all customers who churned" or "all transactions over $500"
you'll ever write starts from this pattern.

## Pandas DataFrames

A `DataFrame` is a table: named columns plus a row index.

```python
import pandas as pd

df = pd.DataFrame({
    "region": ["North", "South", "North", "West", "South"],
    "revenue": [12000, 9500, 15300, 8200, 11100],
    "units": [131, 88, 166, 79, 105],
})
print(df.head())
print(df.dtypes)
```

```text
  region  revenue  units
0  North    12000    131
1  South     9500     88
2  North    15300    166
3   West     8200     79
4  South    11100    105
region       str
revenue    int64
units      int64
dtype: object
```

`.describe()` gives a fast statistical summary of every numeric column —
your first stop on any new dataset:

```python
print(df.describe())
```

```text
           revenue       units
count      5.00000    5.000000
mean   11220.00000  113.800000
std     2708.68972   35.266131
min     8200.00000   79.000000
25%     9500.00000   88.000000
50%    11100.00000  105.000000
75%    12000.00000  131.000000
max    15300.00000  166.000000
```

Creating a new column and filtering rows:

```python
df["revenue_per_unit"] = df["revenue"] / df["units"]
print(df)

print(df[df["revenue"] > 10000])
```

```text
  region  revenue  units  revenue_per_unit
0  North    12000    131         91.603053
1  South     9500     88        107.954545
2  North    15300    166         92.168675
3   West     8200     79        103.797468
4  South    11100    105        105.714286
  region  revenue  units  revenue_per_unit
0  North    12000    131         91.603053
2  North    15300    166         92.168675
4  South    11100    105        105.714286
```

`groupby` — one of the most-used operations in this entire track:

```python
print(df.groupby("region").agg(
    n=("region", "size"),
    total_revenue=("revenue", "sum"),
    avg_rpu=("revenue_per_unit", "mean"),
))
```

```text
        n  total_revenue     avg_rpu
region
North   2          27300   91.885864
South   2          20600  106.834416
West    1           8200  103.797468
```

Read this as: "for each distinct value of `region`, compute these
aggregates." Almost every "break this down by X" question in a real analysis
is a `groupby` with the right aggregation functions.

## Cheat sheet

| Operation | NumPy | pandas |
|---|---|---|
| Shape of the data | `a.shape` | `df.shape` |
| First rows | `a[:5]` | `df.head()` |
| Select column | `M[:, 2]` | `df["col"]` |
| Filter rows | `M[M[:, 0] > 3]` | `df[df["col"] > 3]` |
| Summary stats | `a.mean()`, `a.std()` | `df.describe()` |
| New derived column | — | `df["new"] = df["a"] / df["b"]` |
| Per-group summary | — | `df.groupby("k").agg(...)` |
| Reproducible random draw | `np.random.default_rng(42)` | — |
| To/from NumPy | — | `df.to_numpy()` / `pd.DataFrame(arr)` |

## Exercise

Build a DataFrame of 8 rows with columns `product`, `price`, and
`units_sold` (make up realistic numbers). Using vectorized operations only
(no `for` loops): (1) add a `revenue` column (`price * units_sold`); (2)
filter to products with `revenue` above the dataset's mean revenue; (3)
report total revenue per `product` if you have repeated product names, or
skip step 3 and instead sort the whole table by `revenue` descending.
