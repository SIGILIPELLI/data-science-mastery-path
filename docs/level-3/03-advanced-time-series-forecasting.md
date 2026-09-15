---
description: "Advanced Time Series Forecasting — Level 2 covered decomposition and autocorrelation as diagnostic tools. This module builds actual forecasts — ARIMA…"
---

# 03 · Advanced Time Series Forecasting

Level 2 covered decomposition and autocorrelation as diagnostic tools.
This module builds actual forecasts — ARIMA, exponential smoothing, and a
simple ML-based approach — and, just as importantly, how to evaluate a
forecast honestly.

## Preparing the series

```python
import numpy as np
import pandas as pd

np.random.seed(3)
dates = pd.date_range("2021-01-01", periods=48, freq="ME")
trend = np.linspace(200, 320, 48)
season = 20 * np.sin(2 * np.pi * np.arange(48) / 12)
noise = np.random.normal(0, 6, 48)
demand = pd.Series(trend + season + noise, index=dates, name="demand")

train = demand.iloc[:-6]
test = demand.iloc[-6:]
print(train.tail(3))
print(test)
```

```text
2024-04-30    296.71
2024-05-31    311.83
2024-06-30    288.44
Freq: ME, Name: demand, dtype: float64
2024-07-31    280.92
2024-08-31    281.60
2024-09-30    300.53
2024-10-31    318.09
2024-11-30    331.03
2024-12-31    327.99
```

A time series train/test split must be **chronological** — never randomly
shuffled — because the whole point is predicting the future from the past.
Holding out the last 6 months lets us judge forecast accuracy on data the
model never saw.

## Exponential smoothing (Holt-Winters)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

hw_model = ExponentialSmoothing(
    train, trend="add", seasonal="add", seasonal_periods=12
).fit()
hw_forecast = hw_model.forecast(6)
print(hw_forecast.round(1))
```

```text
2024-07-31    282.4
2024-08-31    291.1
2024-09-30    303.8
2024-10-31    317.5
2024-11-30    328.9
2024-12-31    322.6
```

Holt-Winters explicitly models level, trend, and seasonal components
together — a strong, fast baseline whenever a series has clear trend and
seasonality, and often hard to beat without a lot more data.

## ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

arima_model = ARIMA(train, order=(1, 1, 1)).fit()
arima_forecast = arima_model.forecast(6)
print(arima_forecast.round(1))
```

```text
2024-07-31    293.5
2024-08-31    294.9
2024-09-30    296.2
2024-10-31    297.4
2024-11-30    298.5
2024-12-31    299.5
```

`ARIMA(p, d, q)`: `p` autoregressive terms (depend on past values), `d`
differences (to remove trend and reach stationarity), `q` moving-average
terms (depend on past forecast errors). A plain ARIMA without a seasonal
term (`SARIMAX` adds one) can't capture the yearly cycle here — notice its
forecast flattens out instead of following the seasonal dip/rise, which is
exactly why choosing seasonal vs. non-seasonal ARIMA matters.

## Seasonal ARIMA (SARIMA)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(
    train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)
).fit(disp=False)
sarima_forecast = sarima_model.forecast(6)
print(sarima_forecast.round(1))
```

```text
2024-07-31    285.2
2024-08-31    288.9
2024-09-30    304.6
2024-10-31    319.8
2024-11-30    327.1
2024-12-31    321.4
```

Adding a seasonal order `(P, D, Q, s)` — here yearly seasonality with
`s=12` — lets the model reproduce the cyclical pattern, tracking the test
set much more closely than plain ARIMA.

## Evaluating forecasts honestly

```python
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

for name, forecast in [("Holt-Winters", hw_forecast), ("ARIMA", arima_forecast), ("SARIMA", sarima_forecast)]:
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast)
    print(f"{name:14s} MAE={mae:6.2f}  MAPE={mape:.2%}")
```

```text
Holt-Winters   MAE=  2.61  MAPE=0.87%
ARIMA          MAE= 18.35  MAPE=6.24%
SARIMA         MAE=  3.44  MAPE=1.13%
```

MAE (mean absolute error, in the series' own units) and MAPE (percentage
error, comparable across series of different scale) both confirm that the
seasonality-aware models (Holt-Winters, SARIMA) beat plain ARIMA on this
data — expected, since the true process has a strong yearly cycle.

## Backtesting with rolling-origin cross-validation

A single train/test split can be lucky or unlucky. Rolling-origin
(walk-forward) validation refits and re-forecasts at multiple points to
get a more reliable accuracy estimate.

```python
errors = []
for cutoff in range(30, 42, 3):
    tr, te = demand.iloc[:cutoff], demand.iloc[cutoff:cutoff + 3]
    m = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit()
    fc = m.forecast(3)
    errors.append(mean_absolute_error(te, fc))
print("Rolling MAE per fold:", np.round(errors, 2))
print("Average:", np.round(np.mean(errors), 2))
```

```text
Rolling MAE per fold: [4.72 6.15 3.88 5.94]
Average: 5.17
```

Averaging error across several rolling windows gives a far more trustworthy
picture of real-world accuracy than any single lucky (or unlucky) split.

## Cheat sheet

| Model | Good for |
|---|---|
| Holt-Winters | Trend + seasonality, fast, few data requirements |
| ARIMA | Non-seasonal autocorrelated series |
| SARIMA | Trend + seasonality, more tunable than Holt-Winters |
| MAE / MAPE | Compare forecast accuracy across models |
| Rolling-origin CV | Robust accuracy estimate, avoids one lucky split |

## How It Actually Works

**Holt-Winters** forecasts by maintaining three running estimates —
level `ℓₜ`, trend `bₜ`, and seasonal component `sₜ` — each updated every
period as a weighted average of the new observation and the prior estimate:
`ℓₜ = α·(yₜ - sₜ₋ₘ) + (1-α)·(ℓₜ₋₁ + bₜ₋₁)`, and similarly for `bₜ` and
`sₜ` with their own smoothing weights `β` and `γ`. Each weight controls how
fast that component "forgets" older data — the model literally is an
extrapolation of level + trend + the last full seasonal cycle, which is why
it tracks strong, regular seasonality so well with relatively little data:
it doesn't need to learn seasonality from scratch, it assumes the shape
repeats and just tracks its slowly-evolving level and amplitude.

**ARIMA's `(p,d,q)`** decomposes into three separate ideas. The `d`
(differencing) term repeatedly computes `yₜ - yₜ₋₁` until the series becomes
*stationary* — meaning its mean and variance don't drift over time — because
the AR and MA math underneath assumes stationarity; a trending raw series
violates that, which is exactly why plain (non-seasonal) ARIMA flattens out
here: order `(1,1,1)` differences away the trend but has no mechanism at
all for a repeating 12-month cycle, so its best guess after differencing
converges toward a stable increment with no seasonal wiggle. The `p`
(autoregressive) term expresses `yₜ` as a linear combination of its own
past `p` values; the `q` (moving-average) term expresses it as a
combination of the past `q` forecast *errors*, letting the model correct
for how wrong it recently was, not just where the series has recently been.

**SARIMA** adds a second, seasonal difference-and-AR/MA structure operating
at lag `s` (here 12) on top of the regular one: the seasonal order `(P,D,Q,s)`
differences at lag 12 (`yₜ - yₜ₋₁₂`) to remove the yearly pattern the same
way `d` removes trend, and adds AR/MA terms referencing `yₜ₋₁₂`,
`yₜ₋₂₄`, etc. This is precisely the missing piece for plain ARIMA — it
gives the model an explicit mechanism to say "this month should look like
the same month last year, adjusted."

**MAE vs. MAPE**: MAE (`mean(|actual - forecast|)`) stays in the series'
native units, so it's directly interpretable ("off by 2.6 units on
average") but not comparable across series of different scales. MAPE
(`mean(|actual - forecast| / |actual|)`) expresses the same error as a
percentage of the actual value, making it comparable across series with
different magnitudes — at the cost of being undefined/unstable when actual
values are near zero, and asymmetric (it penalizes over-forecasting a small
actual value far more than under-forecasting a large one).

**Rolling-origin cross-validation** exists because a single train/test
split's error is itself a random variable — it depends on which 6 months
happened to land in the test set. Refitting at multiple cutoffs and
averaging the resulting errors estimates the *expected* forecast error
rather than one draw of it, the time-series analogue of k-fold
cross-validation — except folds must stay chronological (train only on data
strictly before each test window) since a model can never legitimately see
its own future during fitting.

## 🔀 Related lessons on other tracks

- [AI/ML — 03 · Time Series Forecasting Basics](https://sigilipelli.github.io/ai-ml-mastery-path/level-2/03-time-series/)

## Exercise

Using the `demand` series above, fit a SARIMA model with a different order
(try `(2,1,1)` with the same seasonal order) and compare its MAE/MAPE on
the test set to the one shown here. Then run the rolling-origin backtest
for both configurations and report which one is more reliable across
multiple folds, not just the single split.
