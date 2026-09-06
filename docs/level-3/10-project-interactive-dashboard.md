# 10 · Project — Interactive Data Dashboard

This capstone ties together cleaning, aggregation, and visualization from
earlier Level 3 modules into a single interactive dashboard a non-technical
stakeholder could open and explore themselves, using Streamlit (the
lowest-friction way to turn a pandas analysis into a shareable app).

## The dataset and the question

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(7)
n = 2000
dates = pd.date_range("2024-01-01", periods=180, freq="D")

sales = pd.DataFrame({
    "date": rng.choice(dates, n),
    "region": rng.choice(["North", "South", "East", "West"], n),
    "category": rng.choice(["Electronics", "Apparel", "Home", "Grocery"], n),
    "revenue": rng.gamma(shape=2.0, scale=80, size=n).round(2),
    "units": rng.integers(1, 15, n),
})
sales.to_csv("sales.csv", index=False)
```

The dashboard's job: let a regional manager answer "how is revenue trending
by category, filtered to my region and date range?" without writing a line
of code.

## App structure

```text
dashboard/
├── app.py
├── sales.csv
└── requirements.txt
```

```text
# requirements.txt
streamlit==1.36.0
pandas==2.2.2
plotly==5.22.0
```

## Building the app

```python
# app.py
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sales Dashboard", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

df = load_data("sales.csv")

st.title("Regional Sales Dashboard")

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")
    regions = st.multiselect("Region", options=sorted(df["region"].unique()),
                              default=sorted(df["region"].unique()))
    date_range = st.date_input(
        "Date range",
        value=(df["date"].min(), df["date"].max()),
    )

mask = (
    df["region"].isin(regions)
    & (df["date"] >= pd.Timestamp(date_range[0]))
    & (df["date"] <= pd.Timestamp(date_range[1]))
)
filtered = df.loc[mask]

# --- KPI row ---
col1, col2, col3 = st.columns(3)
col1.metric("Total revenue", f"${filtered['revenue'].sum():,.0f}")
col2.metric("Total units", f"{filtered['units'].sum():,}")
col3.metric("Avg order value", f"${filtered['revenue'].mean():,.2f}")

# --- Trend chart ---
daily = filtered.groupby(filtered["date"].dt.to_period("W"))["revenue"].sum().reset_index()
daily["date"] = daily["date"].dt.to_timestamp()
fig_trend = px.line(daily, x="date", y="revenue", title="Weekly revenue trend")
st.plotly_chart(fig_trend, use_container_width=True)

# --- Category breakdown ---
by_cat = filtered.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue")
fig_cat = px.bar(by_cat, x="revenue", y="category", orientation="h", title="Revenue by category")
st.plotly_chart(fig_cat, use_container_width=True)

# --- Raw data ---
with st.expander("View filtered data"):
    st.dataframe(filtered.sort_values("date", ascending=False))
```

```bash
streamlit run app.py
```

`@st.cache_data` matters here: without it, every filter interaction
re-reads and re-parses `sales.csv` from disk, which is fine for 2,000 rows
but would make the app unusable at real dataset sizes — cache the load
step, not the filtering, since filtering is cheap and depends on widget
state.

## Making it feel like a real tool, not a demo

A few small additions separate a "screenshot-able" dashboard from one
people actually return to:

```python
# Guard against an empty filter selection
if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# Let the manager download exactly what they're looking at
st.download_button(
    "Download filtered data as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales.csv",
    mime="text/csv",
)
```

`st.stop()` halts execution of the rest of the script cleanly on an empty
selection, rather than letting downstream `groupby`/`px.line` calls throw
on empty data and print a traceback in front of a non-technical user.

## Deploying it

```bash
# Streamlit Community Cloud: push app.py + requirements.txt to a public
# GitHub repo, then connect the repo at share.streamlit.io — no server
# management required for a small internal tool.
git init
git add app.py requirements.txt sales.csv
git commit -m "Sales dashboard v1"
git push origin main
```

For an internal-only tool behind a company network, `streamlit run app.py
--server.address 0.0.0.0` on a shared machine is often enough; reach for
Community Cloud, Docker + a cloud VM, or an internal PaaS once more than a
handful of people need reliable access.

## Cheat sheet

| Piece | Purpose |
|---|---|
| `st.cache_data` | Avoid re-reading/re-parsing the source file on every interaction |
| Sidebar widgets | Filters that recompute the whole page reactively |
| `st.metric` | At-a-glance KPIs above the fold |
| `st.stop()` | Fail gracefully on empty/invalid filter states |
| `st.download_button` | Let users take the exact filtered slice with them |
| `st.expander` | Hide raw data by default without removing access to it |

## How It Actually Works

This dashboard's structure is a direct, working instance of the rerun
model explained in Module 07: every widget change in the sidebar
(`st.multiselect`, `st.date_input`) triggers a full top-to-bottom
re-execution of `app.py`. `@st.cache_data` on `load_data` is what keeps that
affordable — it hashes the function's arguments (`path`) and, since `path`
never changes between reruns, returns the previously-parsed DataFrame from
memory instead of re-reading and re-parsing `sales.csv` from disk every
single time a filter moves. The `mask`/`filtered` computation deliberately
sits *outside* the cached function precisely because it depends on live
widget state (`regions`, `date_range`) — caching it would either need to key
the cache on every possible filter combination (defeating the purpose) or
silently return stale results for a new filter selection.

The **weekly resample before charting** (`dt.to_period("W")`) is the same
decomposable-aggregation principle from Module 04: summing revenue per
week is a reduction that can be recomputed cheaply on any filtered subset,
which is why filtering-then-aggregating on each rerun is fast enough to
feel instantaneous even though it re-executes on every interaction — 2,000
rows grouped into ~26 weekly buckets is a trivial amount of computation
compared to what a human perceives as lag.

`st.stop()` matters because of how Python (and Streamlit) handle
exceptions during a rerun: if `filtered` is empty, `filtered['revenue'].mean()`
returns `NaN` silently (not an error) but a downstream `px.line` or
`groupby` on a truly empty frame can raise, and an uncaught exception mid-
script would print a Python traceback directly into the deployed app for
whoever's viewing it. Calling `st.stop()` after checking `filtered.empty`
raises Streamlit's own internal `StopException`, which the Streamlit
runtime catches and treats as "the script finished early, cleanly" — the
rest of the script (all the charting code below) simply never executes for
this rerun, and the user sees only the warning message instead of a crash.

`st.download_button`'s `data=filtered.to_csv(index=False).encode("utf-8")`
line does the CSV serialization synchronously on every rerun that reaches
that line — Streamlit needs the actual bytes ready immediately because
there's no server-side "generate on click" callback in this model; the
button's job is purely to hand the browser bytes that were already computed
during the current script execution.

## Exercise

Extend `app.py` with a second tab (`st.tabs`) showing a month-over-month
revenue growth rate per region, computed with `pct_change()` on a monthly
resample. Add a callout (`st.metric` with the `delta` argument) that turns
red when the most recent month's growth is negative.
