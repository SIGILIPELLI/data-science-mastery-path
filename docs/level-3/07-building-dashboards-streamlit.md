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

## Exercise

Extend the dashboard above with a `st.selectbox` letting the user choose
an aggregation level ("Daily", "Weekly", "Monthly"), and resample the
`filtered` DataFrame accordingly before charting (reuse the `resample`
skill from Level 2's time series module). Confirm the chart title and
KPI tiles update correctly when the selection changes.
