---
description: "Time Series Analysis Basics — Time series data — anything indexed by time (daily sales, sensor readings, website traffic) — breaks the 'rows are…"
---

# 06 · Time Series Analysis Basics

Time series data — anything indexed by time (daily sales, sensor readings,
website traffic) — breaks the "rows are independent" assumption most
statistics rely on. This module covers the basic vocabulary and pandas
tooling for working with it.

## Building a proper time index

```python
import pandas as pd
import numpy as np

np.random.seed(1)
dates = pd.date_range("2024-01-01", periods=365, freq="D")
trend = np.linspace(100, 160, 365)
season = 15 * np.sin(2 * np.pi * dates.dayofyear / 7)   # weekly pattern
noise = np.random.normal(0, 5, 365)
sales = pd.Series(trend + season + noise, index=dates, name="sales")
print(sales.head())
```

```text
2024-01-01    103.06
2024-01-02    112.35
2024-01-03    119.53
2024-01-04    118.75
2024-01-05    110.36
Freq: D, Name: sales, dtype: float64
```

A `DatetimeIndex` unlocks date-aware slicing (`sales["2024-03"]` for all of
March), resampling, and rolling windows — always convert a date column
with `pd.to_datetime` and set it as the index before analysis.

## Resampling: changing the time granularity

```python
weekly = sales.resample("W").mean()
print(weekly.head())
```

```text
2024-01-07    111.16
2024-01-14    123.84
2024-01-21    129.15
2024-01-28    132.40
2024-02-04    137.02
Freq: W-SUN, dtype: float64
```

`resample` is groupby for time: `"D"`, `"W"`, `"ME"` (month end), `"QE"`
are the common frequencies. Aggregating up to weekly here averages out the
day-of-week seasonality, making the underlying upward trend obvious.

## Rolling windows: smoothing and moving averages

```python
sales_smoothed = sales.rolling(window=7, center=True).mean()
print(pd.DataFrame({"raw": sales, "7d_avg": sales_smoothed}).iloc[10:15])
```

```text
              raw      7d_avg
2024-01-11  99.42  112.075714
2024-01-12  95.02  113.845714
2024-01-13 103.31  115.313571
2024-01-14 132.66  116.978571
2024-01-15 121.36  118.404286
```

A rolling mean is the simplest way to see the trend through day-to-day
noise. `center=True` aligns the window's label with its midpoint rather
than its right edge — better for visualization, worse for forecasting
(it peeks at "future" values relative to the label).

## Decomposition: trend, seasonality, residual

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(sales, model="additive", period=7)
print(result.trend.dropna().head(3))
print(result.seasonal.head(7).round(2))
```

```text
2024-01-04    111.02
2024-01-05    111.85
2024-01-06    112.71
dtype: float64
2024-01-01    10.83
2024-01-02    14.72
2024-01-03     8.19
2024-01-04    -6.41
2024-01-05   -14.68
2024-01-06   -10.31
2024-01-07    -2.34
Freq: D, dtype: float64
```

`seasonal_decompose` splits a series into **trend** (slow-moving level),
**seasonal** (repeating pattern, here weekly with `period=7`), and
**residual** (what's left, ideally close to noise). This is the fastest
way to check "is there really a weekly pattern here?" before building a
forecasting model around it.

## Autocorrelation: how much does a value depend on its past?

```python
from statsmodels.graphics.tsaplots import acf

values = acf(sales, nlags=10)
print(np.round(values, 2))
```

```text
[1.   0.66 0.34 0.1  0.03 0.05 0.24 0.5  0.31 0.14 0.02]
```

The autocorrelation function (ACF) measures correlation between the series
and a lagged copy of itself. The bump back up at lag 7 (0.24 → 0.5 pattern
returning) confirms the weekly seasonality numerically, not just visually.

## Cheat sheet

| Task | Code |
|---|---|
| Set a date index | `df.set_index(pd.to_datetime(df["date"]))` |
| Change granularity | `series.resample("W").mean()` |
| Smooth with moving average | `series.rolling(7).mean()` |
| Split trend/season/residual | `seasonal_decompose(series, period=7)` |
| Check lag correlation | `acf(series, nlags=10)` |

## How It Actually Works

**Resampling** is a groupby in disguise: `resample("W")` buckets every
timestamp into the week it falls in (using the index's datetime values as
the group key rather than a column), then applies your aggregation
(`.mean()`, `.sum()`) per bucket, exactly like `groupby` did in Module 01.
The difference from a plain groupby is that resample also *fills in* time
buckets that had zero rows (a day with no recorded observations becomes
`NaN` rather than simply missing from the output), because it knows the
full, regular calendar grid the data should live on — a plain groupby has no
such notion of a "gap."

**Rolling windows** compute a statistic over a sliding subsequence: for
`rolling(7).mean()`, the value at position `i` is the mean of positions
`i-6` through `i` inclusive. It's a low-pass filter — averaging cancels
out high-frequency noise (day-to-day fluctuation) while preserving the
lower-frequency trend, because random noise across 7 days tends to partially
offset while a genuine trend pushes every day in a similar direction. The
mechanical cost is that the first 6 values have no full window and come out
`NaN` (or are computed on a partial window if `min_periods` is set), and any
real change in the underlying data takes up to a full window-width to fully
show up in the smoothed line — a rolling average necessarily lags the raw
series.

**Seasonal decomposition** models the series as a combination of trend,
seasonal, and residual components — additively (`y = trend + season +
residual`) when the seasonal swing stays a roughly constant absolute size,
or multiplicatively (`y = trend × season × residual`) when the swing grows
proportionally with the trend level. The trend is extracted first via a
centered moving average with a window equal to the seasonal period (which is
why `period=7` for weekly data), then the seasonal component is estimated by
averaging the detrended values for each position-in-cycle (all Mondays
together, all Tuesdays together, etc.) across every cycle in the data, and
whatever's left after removing both is the residual.

**Autocorrelation** at lag `k` is just the ordinary Pearson correlation
between the series and a copy of itself shifted by `k` steps —
`corr(yₜ, yₜ₋ₖ)`. A value near 0 at most lags with a spike back up at lag 7
is a numerical fingerprint of weekly seasonality: it says "today's value
predicts almost nothing about tomorrow, but it predicts quite a bit about
the same day next week." This is exactly the signal that later forecasting
models (ARIMA's autoregressive term, or a weekly seasonal component) are
built to exploit — the ACF plot is effectively a diagnostic for which lags
are worth feeding a forecasting model as predictors.

## Exercise

Using the `sales` series above, resample to monthly totals with `.resample("ME").sum()`
and plot (or print) the result. Then compute a 30-day rolling standard
deviation — does volatility change over the year, or stay roughly
constant? Explain what that would mean for choosing a forecasting model.
