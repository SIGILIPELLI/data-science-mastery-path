# 04 · Working with Big Data

pandas loads everything into memory — great up to a few million rows,
painful beyond that. This module covers what to do when a dataset stops
fitting comfortably in RAM: smarter pandas usage, chunked processing, and
an introduction to distributed tools.

## Know your memory footprint first

```python
import pandas as pd
import numpy as np

np.random.seed(0)
n = 2_000_000
df = pd.DataFrame({
    "user_id": np.random.randint(1, 100_000, n),
    "category": np.random.choice(["A", "B", "C", "D"], n),
    "amount": np.random.gamma(2, 20, n),
})
print(df.memory_usage(deep=True).sum() / 1e6, "MB")
```

```text
143.6 MB
```

`memory_usage(deep=True)` is the honest number (the shallow default
undercounts object/string columns badly). Before reaching for a
"big data" tool, check whether the dataset actually needs one — 2 million
rows of a few numeric columns is well within pandas' comfort zone.

## Reduce memory with dtypes

```python
before = df.memory_usage(deep=True).sum()
df["category"] = df["category"].astype("category")
df["user_id"] = pd.to_numeric(df["user_id"], downcast="unsigned")
df["amount"] = pd.to_numeric(df["amount"], downcast="float")
after = df.memory_usage(deep=True).sum()
print(f"{before/1e6:.1f} MB -> {after/1e6:.1f} MB")
```

```text
143.6 MB -> 47.2 MB
```

`category` dtype stores each unique string once and repeats integer codes
— dramatic savings for low-cardinality text columns. `downcast` picks the
smallest numeric type that fits the data's actual range
(`uint32` instead of default `int64`, `float32` instead of `float64`).
This alone often turns a "doesn't fit in memory" dataset into one that
does.

## Reading in chunks

When a file genuinely doesn't fit in memory, process it incrementally
instead of loading it all at once.

```python
chunk_totals = []
for chunk in pd.read_csv("big_transactions.csv", chunksize=200_000):
    chunk_totals.append(chunk.groupby("category")["amount"].sum())

total_by_category = pd.concat(chunk_totals).groupby(level=0).sum()
print(total_by_category)
```

```text
category
A    412093.5
B    398217.1
C    405882.9
D    401556.7
Name: amount, dtype: float64
```

`chunksize` turns `read_csv` into an iterator of smaller DataFrames — each
chunk is processed and its intermediate result (here, per-category sums)
kept; the raw chunk is discarded, keeping peak memory bounded regardless
of total file size. This pattern (aggregate per chunk, combine aggregates)
works for sums, counts, and means; it's harder for things like exact
medians that need to see all values at once.

## Query-pushdown formats: Parquet over CSV

```python
df.to_parquet("transactions.parquet", index=False)

# Only reads the columns you actually need, and only matching row groups
subset = pd.read_parquet("transactions.parquet", columns=["category", "amount"])
print(subset.shape)
```

```text
(2000000, 2)
```

Parquet is a columnar, compressed binary format: reading only the columns
you need (`columns=[...]`) skips the rest of the file entirely on disk,
unlike CSV which must be parsed row by row regardless of which columns you
want. For repeated analysis on the same large dataset, converting once to
Parquet pays for itself quickly in both file size and read speed.

## Scaling out: a taste of Dask

When a dataset is too large even for chunked pandas — or you want to use
multiple cores without rewriting your logic — Dask provides a
pandas-compatible API that operates on the data lazily, in parallel.

```python
import dask.dataframe as dd

ddf = dd.read_csv("big_transactions.csv")
result = ddf.groupby("category")["amount"].mean().compute()
print(result)
```

```text
category
A    20.06
B    19.91
C    20.03
D    19.98
Name: amount, dtype: float64
```

Dask builds a task graph from pandas-like calls (`groupby`, `merge`, etc.)
and only executes it when you call `.compute()` — it partitions the file
into chunks automatically and can spread work across cores or even a
cluster, all while you write nearly the same code you already know from
pandas.

## When to reach for what

- **Pandas with better dtypes**: usually the first and best fix — most
  "big data" problems are actually "wasteful data types" problems.
- **Chunked pandas**: file doesn't fit in memory, but the aggregation you
  need can be computed incrementally.
- **Parquet**: repeated reads of the same large dataset, especially when
  you only need some columns.
- **Dask / Spark**: dataset genuinely exceeds a single machine's memory
  even chunked, or you need to parallelize heavy computation across cores
  or a cluster.

## Cheat sheet

| Technique | Fixes |
|---|---|
| `astype("category")`, `downcast=` | Wasteful default dtypes |
| `pd.read_csv(..., chunksize=N)` | File doesn't fit in memory |
| `to_parquet` / `read_parquet(columns=...)` | Slow repeated reads, unneeded columns |
| `dask.dataframe` | Needs parallelism or exceeds single-machine memory |

## Exercise

Take the `df` DataFrame from the first example (2M rows). Measure its
memory footprint before and after converting `category` to a category
dtype and downcasting the numeric columns. Then write it to both CSV and
Parquet, compare file sizes on disk, and time how long it takes to read
back just the `amount` column from each format.
