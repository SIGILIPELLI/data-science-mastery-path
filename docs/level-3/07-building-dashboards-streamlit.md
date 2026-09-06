# 07 · Building Dashboards (Streamlit)

Static reports go stale the moment new data arrives. Streamlit turns a
plain Python script into an interactive web app — no HTML/CSS/JS
required — making it the fastest way for a data scientist to hand
stakeholders a self-serve dashboard.

## A minimal dashboard

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("Regional Sales Dashboard")

@st.cache_data
def load_data():
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    df = pd.DataFrame({
        "date": np.tile(dates, 3),
        "region": np.repeat(["East", "West", "Central"], 180),
        "revenue": np.random.gamma(4, 50, 540),
    })
    return df

df = load_data()
st.dataframe(df.head())
```

Running `streamlit run app.py` starts a local server and opens the app in
a browser; every widget interaction reruns the script top to bottom.
`@st.cache_data` prevents expensive operations (loading, cleaning) from
re-running on every single interaction — without it, this toy example
would still be fast, but a real database query or model load would make
the app feel sluggish.

## Adding interactive filters

```python
region_filter = st.multiselect(
    "Region", options=df["region"].unique(), default=df["region"].unique()
)
date_range = st.date_input(
    "Date range", value=(df["date"].min(), df["date"].max())
)

filtered = df[
    df["region"].isin(region_filter)
    & (df["date"] >= pd.Timestamp(date_range[0]))
    & (df["date"] <= pd.Timestamp(date_range[1]))
]
st.write(f"Showing {len(filtered)} rows")
```

Widgets return their current value directly as a Python variable —
`st.multiselect` returns a list, `st.date_input` a tuple of dates. There's
no callback wiring to write: filter the DataFrame with the returned
values, and any downstream chart or table using `filtered` updates
automatically on the next rerun.

## Layout: columns and metrics

```python
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${filtered['revenue'].sum():,.0f}")
col2.metric("Avg Daily Revenue", f"${filtered.groupby('date')['revenue'].sum().mean():,.0f}")
col3.metric("Regions Shown", filtered["region"].nunique())
```

`st.columns(n)` splits the page into side-by-side sections;
`st.metric` renders a label, a big number, and (optionally) a delta —
the standard KPI-tile look expected at the top of a dashboard.

## Charts

```python
import plotly.express as px

daily = filtered.groupby(["date", "region"])["revenue"].sum().reset_index()
fig = px.line(daily, x="date", y="revenue", color="region", title="Daily Revenue by Region")
st.plotly_chart(fig, use_container_width=True)

by_region = filtered.groupby("region")["revenue"].sum().sort_values(ascending=False)
st.bar_chart(by_region)
```

`st.plotly_chart` embeds a fully interactive Plotly figure (hover tooltips,
zoom, legend toggling) with one line; `st.bar_chart` is a quick built-in
alternative when you don't need Plotly's extra interactivity.

## Sidebar and page organization

```python
with st.sidebar:
    st.header("Filters")
    st.write("Use the controls above to narrow down the data.")
    threshold = st.slider("Highlight days above revenue", 0, 1000, 300)

high_days = filtered[filtered["revenue"] > threshold]
st.subheader(f"Days above ${threshold}")
st.dataframe(high_days.sort_values("revenue", ascending=False))
```

Moving controls into `st.sidebar` keeps the main area for content and
gives the dashboard a familiar "filters on the left, results on the
right" layout stakeholders already expect from BI tools.

## Deploying

```bash
# requirements.txt
streamlit
pandas
plotly

# Then, from Streamlit Community Cloud or any server:
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

For a stakeholder-facing dashboard, Streamlit Community Cloud (free, git-
push-to-deploy) is usually the fastest path from script to shareable URL;
for internal data behind a firewall, running it on an internal server or
container is more common — either way, keep secrets (database
credentials) in `st.secrets` or environment variables, never hardcoded in
the script.

## Cheat sheet

| Task | Code |
|---|---|
| Cache expensive loads | `@st.cache_data` above a function |
| Filter widget | `st.multiselect(...)`, `st.date_input(...)` |
| KPI tile | `st.metric(label, value, delta=)` |
| Interactive chart | `st.plotly_chart(fig, use_container_width=True)` |
| Sidebar controls | `with st.sidebar: ...` |
| Run locally | `streamlit run app.py` |

## How It Actually Works

Streamlit's entire programming model rests on one mechanical rule: **any
widget interaction re-executes the whole script from top to bottom** — there
is no callback graph or partial re-render like a traditional web framework.
When you move the `region_filter` multiselect, Streamlit doesn't call some
registered "on change" function; it reruns `app.py` in its entirety, and on
this rerun `st.multiselect` immediately returns the new selection (Streamlit
tracks each widget's current value keyed by its position/label in the
script and feeds it back in). This is what makes the code read like a
plain, linear script with no event wiring — the "reactivity" is really just
"rerun everything, cheaply."

That rerun-everything model is also exactly why `@st.cache_data` exists and
matters. Without it, `load_data()` — or a real database query — would
re-execute on *every* widget nudge, even though its result never changed.
`st.cache_data` works by hashing the function's arguments (and its source
code) to build a cache key; on a rerun with the same arguments, Streamlit
skips calling the function and returns the previously stored result
directly, turning an O(script reruns × expensive work) cost into
O(expensive work once) plus cheap cache lookups thereafter. The cache is
invalidated automatically if the function's code or arguments change,
which is why it's safe to leave in place during development.

**Session state persistence** (implicit here, since widget values must
survive across reruns) is handled by Streamlit associating each widget
instance with a stable key derived from its position and arguments in the
script, storing its current value server-side between reruns — this is why
two widgets with the exact same label and type can silently collide unless
given an explicit `key=`, since Streamlit has no other way to tell them
apart across reruns.

**Deployment's `--server.address 0.0.0.0`** flag matters mechanically
because Streamlit's dev server binds to `localhost` (`127.0.0.1`) by
default, which only accepts connections originating from the same machine;
binding to `0.0.0.0` tells the OS to accept connections on *any* network
interface, which is what makes the app reachable from other machines at
all (a container's own internal network, or the public internet, depending
on what else fronts it) — without that flag, a dashboard "deployed" to a
server would still only be visible to someone logged into that server
directly.

## Exercise

Extend the dashboard above with a `st.selectbox` letting the user choose
an aggregation level ("Daily", "Weekly", "Monthly"), and resample the
`filtered` DataFrame accordingly before charting (reuse the `resample`
skill from Level 2's time series module). Confirm the chart title and
KPI tiles update correctly when the selection changes.
