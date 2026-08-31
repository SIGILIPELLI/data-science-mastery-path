# 08 · SQL for Data Scientists

Most production data lives in a database, not a CSV. This module covers
the SQL a data scientist uses daily — enough to pull and shape your own
data instead of waiting on a data engineer — using Python's `sqlite3` so
every example runs standalone.

## Setting up a sample database

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect(":memory:")
orders = pd.DataFrame({
    "order_id": range(1, 7),
    "customer_id": [1, 1, 2, 3, 2, 1],
    "amount": [50, 120, 75, 200, 40, 90],
    "order_date": pd.to_datetime(
        ["2024-01-05", "2024-02-10", "2024-01-20", "2024-03-01", "2024-02-15", "2024-03-10"]
    ),
})
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "name": ["Priya", "Tom", "Aisha"],
    "signup_date": pd.to_datetime(["2023-11-01", "2023-12-15", "2024-01-01"]),
})
orders.to_sql("orders", conn, index=False)
customers.to_sql("customers", conn, index=False)
```

## SELECT, WHERE, ORDER BY

```python
q = """
SELECT order_id, customer_id, amount
FROM orders
WHERE amount > 60
ORDER BY amount DESC
"""
print(pd.read_sql(q, conn))
```

```text
   order_id  customer_id  amount
0         4            3     200
1         2            1     120
2         6            1      90
3         3            2      75
```

`pd.read_sql` runs the query and returns a DataFrame directly — the
standard bridge between a database and a pandas analysis.

## JOIN: combining tables

```python
q = """
SELECT c.name, o.order_id, o.amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
ORDER BY c.name, o.order_id
"""
print(pd.read_sql(q, conn))
```

```text
    name  order_id  amount
0  Aisha         4     200
1   Priya        1      50
2   Priya        2     120
3   Priya        6      90
4    Tom         3      75
5    Tom         5      40
```

`JOIN` (an inner join by default) is SQL's `merge` — it matches rows across
tables on the given key. Use `LEFT JOIN` to keep every row from the first
table even without a match, exactly like pandas' `how="left"`.

## GROUP BY and aggregates

```python
q = """
SELECT c.name, COUNT(*) AS n_orders, SUM(o.amount) AS total_spent
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.name
HAVING SUM(o.amount) > 100
ORDER BY total_spent DESC
"""
print(pd.read_sql(q, conn))
```

```text
    name  n_orders  total_spent
0   Priya         3          260
1   Aisha         1          200
```

`WHERE` filters rows before grouping; `HAVING` filters *after* aggregation
(you can't write `WHERE SUM(amount) > 100` — the aggregate doesn't exist
yet at that stage of the query).

## Window functions: per-row context without collapsing rows

```python
q = """
SELECT order_id, customer_id, amount,
       SUM(amount) OVER (PARTITION BY customer_id) AS customer_total,
       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS order_seq
FROM orders
ORDER BY customer_id, order_seq
"""
print(pd.read_sql(q, conn))
```

```text
   order_id  customer_id  amount  customer_total  order_seq
0         1            1      50             260          1
1         2            1     120             260          2
2         6            1      90             260          3
3         3            2      75             115          1
4         5            2      40             115          2
5         4            3     200             200          1
```

`OVER (PARTITION BY ...)` is the SQL equivalent of pandas' `groupby(...).transform(...)` —
it computes a per-group value without collapsing rows, so you keep
row-level detail *and* group context in the same result. `ROW_NUMBER()`
numbers rows within each partition — handy for "each customer's first
order" (`WHERE order_seq = 1`).

## CTEs: naming intermediate steps

```python
q = """
WITH customer_totals AS (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT c.name, ct.total_spent
FROM customer_totals ct
JOIN customers c ON c.customer_id = ct.customer_id
WHERE ct.total_spent > 100
"""
print(pd.read_sql(q, conn))
```

```text
    name  total_spent
0  Priya          260
1  Aisha          200
```

A `WITH` clause (Common Table Expression) names an intermediate query so
the final `SELECT` reads top-to-bottom like a pipeline — much more
maintainable than nesting subqueries, and the standard style for anything
non-trivial.

## Cheat sheet

| Task | SQL |
|---|---|
| Filter rows | `WHERE condition` |
| Filter after aggregation | `HAVING condition` |
| Combine tables | `JOIN ... ON key` / `LEFT JOIN` |
| Per-group value, keep rows | `SUM(x) OVER (PARTITION BY k)` |
| Rank/number within group | `ROW_NUMBER() OVER (PARTITION BY k ORDER BY col)` |
| Named intermediate query | `WITH name AS (...) SELECT ...` |

## Exercise

Using the `orders`/`customers` tables above, write one query with a CTE
that computes each customer's average order amount, then a final SELECT
that returns only customers whose average order is above the overall
average across all customers. (Hint: you'll need a second CTE or a
subquery for the overall average.)
