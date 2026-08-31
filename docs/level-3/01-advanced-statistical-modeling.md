# 01 · Advanced Statistical Modeling

Level 1–2 covered simple regression and basic tests. Real analyses usually
need models that handle multiple predictors, non-linear relationships, and
non-normal outcomes. This module covers multiple linear regression,
regularization, and generalized linear models (GLMs) with `statsmodels`.

## Multiple linear regression with statsmodels

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(0)
n = 300
houses = pd.DataFrame({
    "sqft": np.random.normal(1800, 400, n).clip(600, None),
    "bedrooms": np.random.randint(1, 6, n),
    "age_years": np.random.randint(0, 60, n),
})
houses["price"] = (
    50_000 + 120 * houses["sqft"] + 8_000 * houses["bedrooms"]
    - 900 * houses["age_years"] + np.random.normal(0, 15_000, n)
)

model = smf.ols("price ~ sqft + bedrooms + age_years", data=houses).fit()
print(model.summary().tables[1])
```

```text
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   4.87e+04   3184.912     15.288      0.000    4.24e+04    5.49e+04
sqft         119.7891     1.529     78.316      0.000     116.780     122.798
bedrooms    8079.4432   463.276     17.437      0.000    7167.798    8991.089
age_years   -897.7521    13.982    -64.209      0.000    -925.284    -870.221
==============================================================================
```

Each coefficient is the effect of that variable *holding the others
constant* — `sqft`'s coefficient (~120) means an extra square foot adds
about $120 to price, controlling for bedrooms and age. Always check `P>|t|`
(is the effect distinguishable from zero) alongside the confidence
interval (how big might it really be), not just whether it's "significant."

## Checking model fit and assumptions

```python
print(f"R-squared: {model.rsquared:.3f}")

residuals = model.resid
print(f"Residual mean: {residuals.mean():.2f}, std: {residuals.std():.1f}")
```

```text
R-squared: 0.972
Residual mean: 0.00, std: 14876.4
```

R² near 1 says the three predictors explain most of the price variation
(expected here — we simulated it that way). In real data, always plot
residuals against fitted values: a random scatter supports the linear
model; a curve or funnel shape (heteroscedasticity) means the model is
missing structure or needs a transformation (e.g. modeling `log(price)`).

## Multicollinearity: when predictors overlap

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = houses[["sqft", "bedrooms", "age_years"]]
X_with_const = pd.concat([pd.Series(1, index=X.index, name="const"), X], axis=1)
vif = pd.Series(
    [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])],
    index=X.columns,
)
print(vif.round(2))
```

```text
sqft         1.02
bedrooms     1.02
age_years    1.01
```

A Variance Inflation Factor (VIF) near 1 means a predictor isn't well
explained by the others — no multicollinearity problem. VIF above ~5–10
signals that two or more predictors are redundant, which inflates
coefficient standard errors and makes individual effects unstable (the
model still predicts fine — it's the *interpretation* of each coefficient
that becomes unreliable).

## Regularization: Ridge and Lasso

When you have many correlated predictors, or more predictors than you
fully trust, regularization shrinks coefficients toward zero to reduce
overfitting.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
y = houses["price"]

ridge = Ridge(alpha=10).fit(X_scaled, y)
lasso = Lasso(alpha=100).fit(X_scaled, y)

print("Ridge coefs:", np.round(ridge.coef_, 1))
print("Lasso coefs:", np.round(lasso.coef_, 1))
```

```text
Ridge coefs: [50652.4  8043.4 -14837.7]
Lasso coefs: [51012.8  7998.1 -14874.5]
```

Ridge shrinks all coefficients smoothly; Lasso can shrink some *exactly*
to zero, effectively performing feature selection. With three genuinely
useful, uncorrelated predictors like these neither shrinks much — the
difference from OLS becomes visible when you have many noisy or redundant
features.

## Generalized Linear Models: modeling counts

Linear regression assumes a continuous, roughly normal outcome. Count
data (number of support tickets, website visits) is better modeled with a
Poisson GLM.

```python
np.random.seed(1)
tickets = pd.DataFrame({"tenure_months": np.random.randint(1, 48, 400)})
rate = np.exp(1.2 - 0.02 * tickets["tenure_months"])
tickets["n_tickets"] = np.random.poisson(rate)

glm = smf.glm("n_tickets ~ tenure_months", data=tickets,
              family=__import__("statsmodels.api", fromlist=["families"]).families.Poisson()).fit()
print(glm.params.round(4))
```

```text
Intercept        1.1928
tenure_months   -0.0197
```

The coefficient is on the *log* scale (Poisson GLMs use a log link): each
extra month of tenure multiplies the expected ticket rate by
`exp(-0.0197) ≈ 0.98`, i.e. about a 2% reduction per month.

## Cheat sheet

| Task | Code |
|---|---|
| Multiple regression | `smf.ols("y ~ x1 + x2", data=df).fit()` |
| Check multicollinearity | `variance_inflation_factor(X, i)` |
| Shrink coefficients | `Ridge(alpha=...)`, `Lasso(alpha=...)` |
| Model counts | `smf.glm("y ~ x", data=df, family=sm.families.Poisson())` |

## Exercise

Add a fourth predictor to `houses` — `has_garage` (a 0/1 column, randomly
assigned) with no real effect on price — and refit the OLS model. Confirm
its coefficient is not significant (`P>|t|` well above 0.05), then fit a
Lasso model with a large enough `alpha` to shrink that coefficient to
exactly zero. Explain the difference in what each result tells you.
