# 08 · Working with Real-World Messy Datasets

Modules 03–07 introduced cleaning and analysis techniques individually. This
module combines them on one dataset that looks like a genuine export from an
order system — inconsistent country names, missing customers, a blank
amount, and a negative refund row hiding among normal orders.

## Loading the raw data

```python
import io
import pandas as pd

csv_text = """order_id,customer,amount,country,notes
1001,Acme Co,1200.50,USA,
1002,Beta LLC,,CA,partial refund
1003,Acme Co,980.00,usa,
1004,Gamma Inc,2450.75,United States,rush order
1005,,150.00,USA,walk-in
1006,Beta LLC,300.00,Canada,
1007,Delta,-50.00,USA,refund
"""
df = pd.read_csv(io.StringIO(csv_text))
print(df)
print(df.dtypes)
```

```text
   order_id   customer   amount        country           notes
0      1001    Acme Co  1200.50            USA             NaN
1      1002   Beta LLC      NaN             CA  partial refund
2      1003    Acme Co   980.00            usa             NaN
3      1004  Gamma Inc  2450.75  United States      rush order
4      1005        NaN   150.00            USA         walk-in
5      1006   Beta LLC   300.00         Canada             NaN
6      1007      Delta   -50.00            USA          refund
order_id      int64
customer        str
amount      float64
country         str
notes           str
dtype: object
```

In real order data, `pd.read_csv` reading `amount` as `float64` despite one
blank cell is itself a signal — pandas silently converted the missing value
to `NaN` and kept the column numeric. If even one cell had contained text
like `"N/A"` where a number belonged, the whole column would have been read
as `object`/`str` instead, silently breaking every downstream calculation.
Always check `.dtypes` right after loading, before anything else.

## Standardizing inconsistent categories

`country` has four different spellings for what are really two countries —
`"USA"`, `"usa"`, `"United States"` all mean the same thing:

```python
country_map = {"usa": "USA", "united states": "USA", "canada": "CA", "ca": "CA"}
df["country_clean"] = (
    df["country"].str.strip().str.lower().map(country_map).fillna(df["country"])
)
print(df[["country", "country_clean"]])
```

```text
         country country_clean
0            USA           USA
1             CA            CA
2            usa           USA
3  United States           USA
4            USA           USA
5         Canada            CA
6            USA           USA
```

The pattern — lowercase/strip, then map known variants to a canonical
value, then `.fillna()` back to the original for anything the map didn't
cover — is the general-purpose way to clean any free-text category column
without needing to enumerate every possible input up front.

## Missing customer names

```python
print(df["customer"].isna().sum())      # 1
df["customer"] = df["customer"].fillna("Unknown")
```

```text
1
```

One order has no customer name (`order_id 1005`, a walk-in per its
`notes`). Filling with an explicit `"Unknown"` label — rather than dropping
the row — keeps the revenue in any downstream `.sum()`, while making the gap
visible rather than silently averaged away like a numeric `fillna(mean)`
would.

## Finding rows that don't belong

```python
negative = df[df["amount"] < 0]
print(negative)
```

```text
   order_id customer  amount country   notes country_clean
6      1007    Delta   -50.0     USA  refund             USA
```

A negative `amount` in an "orders" table is either a refund (legitimate,
but arguably belongs in a separate refunds analysis) or a data error. The
`notes` column confirms it's a refund here — the kind of cross-column check
that catches problems a single-column scan would miss. Whether to exclude
refunds from a "total sales" figure or report them separately is a judgment
call that belongs in your write-up, not something to decide silently.

## Handling the missing amount and finalizing

```python
print(df["amount"].isna().sum())        # 1

df_clean = df.dropna(subset=["amount"]).copy()
print(df_clean.groupby("country_clean")["amount"].agg(["count", "sum", "mean"]).round(2))
```

```text
1
               count      sum    mean
country_clean
CA                 1   300.00  300.00
USA                5  4731.25  946.25
```

Here `dropna(subset=["amount"])` was chosen over filling with a mean/median
because a missing *transaction amount* on a "partial refund" order is
ambiguous enough (was it $0? was it not yet recorded?) that guessing a
number would be worse than excluding the row and noting the exclusion. This
is the same "no single right answer, but you must document the choice"
principle from Module 03 — applied here to a case where dropping, not
filling, was the more honest option.

## The general playbook

Every messy real-world dataset responds to the same sequence, in this order:

1. Load it, immediately check `.dtypes` — did anything read as the wrong type?
2. Standardize categorical spelling/casing before doing anything else with
   those columns (Module 03).
3. Check for and decide on missing values *per column*, based on what that
   column means (Modules 03, 08) — don't apply one blanket rule to
   everything.
4. Look for rows that are structurally different (refunds, cancellations,
   test data, duplicates) and decide whether they belong in your main
   analysis or a separate one.
5. Only then run your EDA/statistics/regression (Modules 04, 06, 07) — on
   data you've actually verified, not assumed.

## Cheat sheet

| Problem | Fix |
|---|---|
| Inconsistent category spelling | `.str.strip().str.lower().map({...}).fillna(original)` |
| Missing categorical value | `.fillna("Unknown")` — keep it visible, don't drop silently |
| Ambiguous missing numeric value | Consider `dropna()` over guessing; document the choice |
| Rows that don't belong (refunds, tests) | Filter and analyze separately, don't silently exclude *or* silently include |
| Wrong dtype on load | Check `.dtypes` immediately after `pd.read_csv` |

## Exercise

Add two more rows to `csv_text`: one with `amount = "unknown"` (a string in
a numeric column) and one exact duplicate of an existing order. Reload the
CSV, and answer: (1) what does `.dtypes` show for `amount` now, and why;
(2) how would you fix it back to numeric; (3) how do you detect and remove
the duplicate using the technique from Module 03.
