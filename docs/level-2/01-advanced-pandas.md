# 01 · Advanced pandas (groupby, merge, pivot)

Level 1 covered pandas basics — loading data, selecting columns, filtering
rows. Real analysis work lives in three operations you'll use every single
day: **groupby** (aggregate within categories), **merge** (combine tables),
and **pivot** (reshape long data into wide summaries).

## groupby: split, apply, combine

```python
import pandas as pd

orders = pd.DataFrame({
    "region": ["East", "East", "West", "West", "East", "West"],
    "product": ["Widget", "Gadget", "Widget", "Gadget", "Widget", "Widget"],
    "revenue": [120, 340, 90, 410, 150, 200],
    "units":   [4, 2, 3, 3, 5, 6],
})

by_region = orders.groupby("region")["revenue"].agg(["sum", "mean", "count"])
print(by_region)
```

```text
        sum        mean  count
region
East    610  203.333333      3
West    700  233.333333      3
```

`groupby` splits the DataFrame into groups by the key column, applies an
aggregation to each group independently, then stitches the results back
into one table — pandas does the loop for you, in compiled C, which is why
it's dramatically faster than a Python `for` loop over rows.

Grouping by more than one key gives a hierarchical (MultiIndex) result:

```python
by_both = orders.groupby(["region", "product"]).agg(
    total_revenue=("revenue", "sum"),
    avg_units=("units", "mean"),
)
print(by_both)
```

```text
                 total_revenue  avg_units
region product
East   Gadget              340        2.0
       Widget              270        4.5
West   Gadget              410        3.0
       Widget              290        4.5
```

The named-aggregation syntax (`total_revenue=("revenue", "sum")`) is the
modern, readable way to compute several differently-named aggregates in one
call — prefer it over renaming columns afterward.

## transform: group stats broadcast back to every row

Sometimes you don't want a summary table — you want each row annotated with
its group's statistic (e.g. "revenue as a % of this region's total"):

```python
orders["region_total"] = orders.groupby("region")["revenue"].transform("sum")
orders["pct_of_region"] = (orders["revenue"] / orders["region_total"] * 100).round(1)
print(orders[["region", "revenue", "region_total", "pct_of_region"]])
```

```text
  region  revenue  region_total  pct_of_region
0   East      120           610           19.7
1   East      340           610           55.7
2   West       90           700           12.9
3   West      410           700           58.6
4   East      150           610           24.6
5   West      200           700           28.6
```

`transform` returns a Series the same length as the input, aligned by
index — unlike `agg`, which collapses each group to one row. This is the
standard pattern for "what share of the group does this row represent?"

## merge: combining tables on a key

```python
customers = pd.DataFrame({
    "region": ["East", "West", "North"],
    "manager": ["Priya", "Tom", "Aisha"],
})

merged = orders.merge(customers, on="region", how="left")
print(merged[["region", "product", "revenue", "manager"]])
```

```text
  region product  revenue manager
0   East  Widget      120   Priya
1   East  Gadget      340   Priya
2   West  Widget       90     Tom
3   West  Gadget      410     Tom
4   East  Widget      150   Priya
5   West  Widget      200     Tom
```

`how="left"` keeps every row from `orders` (the left table) and attaches
matching data from `customers`; a region with no match would get `NaN`.
`"North"` from `customers` disappears entirely because no order matches
it — that's the defining behavior of a left join. Use `how="inner"` to keep
only rows that match on both sides, `how="outer"` to keep everything from
both sides (filling unmatched cells with `NaN`), and always check
`len(merged)` after a merge — an unexpected row-count change usually means
your join key has duplicates you didn't account for.

## pivot_table: long data to a wide summary grid

```python
pivot = orders.pivot_table(
    index="region", columns="product", values="revenue", aggfunc="sum", fill_value=0
)
print(pivot)
```

```text
product  Gadget  Widget
region
East        340     420
West        410     290
```

`pivot_table` is `groupby` + reshape in one step: it groups by the `index`
and `columns` keys simultaneously and lays the aggregated values out as a
grid, which is exactly the shape you want for a summary table or a heatmap.
`fill_value=0` replaces combinations with no data (rather than `NaN`) —
appropriate here since "no orders" really does mean zero revenue.

## melt: wide back to long

The inverse of pivoting — useful when a wide table needs to go back to
one-row-per-observation for further analysis or plotting:

```python
wide = pivot.reset_index()
long = wide.melt(id_vars="region", var_name="product", value_name="revenue")
print(long.sort_values("region").reset_index(drop=True))
```

```text
  region product  revenue
0   East  Gadget      340
1   East  Widget      420
2   West  Gadget      410
3   West  Widget      290
```

## Cheat sheet

| Task | Method |
|---|---|
| Aggregate within groups | `df.groupby("k").agg(name=("col", "sum"))` |
| Broadcast a group stat to every row | `df.groupby("k")["col"].transform("sum")` |
| Combine two tables on a key | `df.merge(other, on="k", how="left")` |
| Long → wide summary grid | `df.pivot_table(index=, columns=, values=, aggfunc=)` |
| Wide → long | `df.melt(id_vars=, var_name=, value_name=)` |

## Exercise

Using the `orders` DataFrame above, compute each product's revenue as a
percentage of its *own* total across both regions (not the region total).
Then build a `pivot_table` showing average `units` per order by region and
product, with missing combinations filled with `0`. Which region/product
combination has the highest average units per order?
