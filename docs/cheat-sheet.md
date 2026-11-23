---
title: "Data Science Cheat Sheet: Quick Reference, Beginner to Advanced"
description: "A free Data Science cheat sheet: 37 quick-reference tables from every level of the Data Science Mastery Path, each linked to its full lesson."
---

# Data Science Cheat Sheet

Quick-reference tables from **37 lessons** of the free Data Science Mastery Path, in course order from beginner to advanced. Each section links to the full lesson with explanations and runnable examples.

## Level 1 · Entry

### [01 · What Is Data Science?](level-1/01-what-is-data-science.md)

| Concept | One-line definition |
|---|---|
| Data science | Turning raw data into a decision, using stats + code + domain sense |
| Data engineering | Building the pipelines that make data reliably available |
| ML engineering | Deploying and operating models in production at scale |
| EDA | Exploratory Data Analysis — looking at data before modeling it |
| The 80% rule | Cleaning/understanding data usually dominates project time |
| "So what?" test | Every finding should trace to a decision or action |

### [02 · Python for Data Science](level-1/02-python-for-data-science.md)

| Operation | NumPy | pandas |
|---|---|---|
| Shape of the data | `a.shape` | `df.shape` |
| First rows | `a[:5]` | `df.head()` |
| Select column | `M[:, 2]` | `df["col"]` |
| Filter rows | `M[M[:, 0] > 3]` | `df[df["col"] > 3]` |
| Summary stats | `a.mean()`, `a.std()` | `df.describe()` |
| New derived column | — | `df["new"] = df["a"] / df["b"]` |
| Per-group summary | — | `df.groupby("k").agg(...)` |
| Reproducible random draw | `np.random.default_rng(42)` | — |
| To/from NumPy | — | `df.to_numpy()` / `pd.DataFrame(arr)` |

### [03 · Data Cleaning &amp; Wrangling](level-1/03-data-cleaning-wrangling.md)

| Task | Code |
|---|---|
| Count missing per column | `df.isna().sum()` |
| Find duplicate rows | `df.duplicated(subset=["col"])` |
| Drop duplicates | `df.drop_duplicates(subset=["col"], keep="first")` |
| Normalize text | `df["c"].str.strip().str.lower()` |
| Parse mixed-format dates | `pd.to_datetime(df["c"], format="mixed")` |
| Fill missing numeric | `df["c"].fillna(df["c"].median())` |
| Rename columns | `df.rename(columns={"old": "new"})` |
| Check types after cleaning | `df.dtypes` |

### [04 · Exploratory Data Analysis (EDA)](level-1/04-exploratory-data-analysis.md)

| Question | Code |
|---|---|
| What am I looking at? | `df.shape`, `df.dtypes`, `df.head()` |
| Summary stats | `df.describe()` |
| Outlier bounds (IQR rule) | `q1 - 1.5*iqr`, `q3 + 1.5*iqr` |
| Category frequencies | `df["c"].value_counts(normalize=True)` |
| Compare groups | `df.groupby("g")[["c1","c2"]].mean()` |
| Linear relationships | `df.corr(numeric_only=True)` |

### [05 · Data Visualization Basics](level-1/05-data-visualization-basics.md)

| Task | Code |
|---|---|
| Histogram | `ax.hist(df["col"], bins=20)` |
| Scatter | `ax.scatter(df["x"], df["y"], alpha=0.5)` |
| Bar of a groupby | `df.groupby("g")["v"].mean().plot(kind="bar")` |
| Box plot by category | `sns.boxplot(data=df, x="cat", y="num")` |
| Always label axes | `ax.set_xlabel(...)`, `ax.set_ylabel(...)` |
| Save a figure | `fig.savefig("name.png", dpi=150)` |

### [06 · Statistics Fundamentals](level-1/06-statistics-fundamentals.md)

| Concept | Code | What it tells you |
|---|---|---|
| Sample mean/std | `x.mean()`, `x.std(ddof=1)` | Center and spread of your data |
| t-test | `stats.ttest_ind(a, b)` | Is a difference between two groups likely real? |
| p-value | second output of `ttest_ind` | P(data this extreme \| no true difference) |
| Confidence interval | `stats.t.interval(0.95, df, loc, scale)` | Plausible range for the true value |
| p-hacking | — | Testing many slices and reporting only the "hits" |
| Bonferroni fix | `alpha / n_tests` | Adjusted threshold for multiple comparisons |

### [07 · Correlation &amp; Basic Regression](level-1/07-correlation-basic-regression.md)

| Task | Code |
|---|---|
| Correlation + significance | `stats.pearsonr(x, y)` |
| Fit a regression line | `sm.OLS(y, sm.add_constant(x)).fit()` |
| Slope/intercept | `model.params` |
| Variance explained | `model.rsquared` |
| Are coefficients significant? | `model.pvalues` |
| Check for Simpson's paradox | `df.groupby([confounder, group])[outcome].mean()` vs. aggregate |

### [08 · Working with Real-World Messy Datasets](level-1/08-messy-datasets.md)

| Problem | Fix |
|---|---|
| Inconsistent category spelling | `.str.strip().str.lower().map({...}).fillna(original)` |
| Missing categorical value | `.fillna("Unknown")` — keep it visible, don't drop silently |
| Ambiguous missing numeric value | Consider `dropna()` over guessing; document the choice |
| Rows that don't belong (refunds, tests) | Filter and analyze separately, don't silently exclude *or* silently include |
| Wrong dtype on load | Check `.dtypes` immediately after `pd.read_csv` |

### [09 · Communicating Findings](level-1/09-communicating-findings.md)

| Task | Guidance |
|---|---|
| Bar/column chart axis | Always start at 0 |
| Line chart showing magnitude of change | Start at 0 unless zooming is the explicit point, and label clearly |
| Leading with the finding | State it in one sentence before showing any chart |
| Reporting a %, always also report | The absolute numbers behind it |
| Claiming a relationship | Say "associated with," never "causes," unless from a controlled experiment |
| One chart, one point | Split a chart that needs a paragraph to explain |

## Level 2 · Intermediate

### [01 · Advanced pandas (groupby, merge, pivot)](level-2/01-advanced-pandas.md)

| Task | Method |
|---|---|
| Aggregate within groups | `df.groupby("k").agg(name=("col", "sum"))` |
| Broadcast a group stat to every row | `df.groupby("k")["col"].transform("sum")` |
| Combine two tables on a key | `df.merge(other, on="k", how="left")` |
| Long → wide summary grid | `df.pivot_table(index=, columns=, values=, aggfunc=)` |
| Wide → long | `df.melt(id_vars=, var_name=, value_name=)` |

### [04 · Feature Engineering Basics](level-2/04-feature-engineering.md)

| Task | Code |
|---|---|
| Nominal category → dummy columns | `pd.get_dummies(df, columns=[...])` |
| Ordered category → rank number | `series.map({"low": 0, "high": 1})` |
| Standardize numeric features | `StandardScaler().fit_transform(X)` |
| Continuous → buckets | `pd.cut(series, bins=[...], labels=[...])` |
| Date difference → numeric | `(date_a - date_b).dt.days` |

### [05 · Intro to Machine Learning for Data Science](level-2/05-intro-machine-learning.md)

| Task | Code |
|---|---|
| Split data | `train_test_split(X, y, test_size=.25, stratify=y)` |
| Scale + model in one object | `make_pipeline(StandardScaler(), LogisticRegression())` |
| Fit / predict | `model.fit(X_train, y_train)`, `model.predict(X_test)` |
| Precision/recall/F1 | `classification_report(y_test, preds)` |
| Feature importance | `model.feature_importances_` (tree models) |

### [06 · Time Series Analysis Basics](level-2/06-time-series-basics.md)

| Task | Code |
|---|---|
| Set a date index | `df.set_index(pd.to_datetime(df["date"]))` |
| Change granularity | `series.resample("W").mean()` |
| Smooth with moving average | `series.rolling(7).mean()` |
| Split trend/season/residual | `seasonal_decompose(series, period=7)` |
| Check lag correlation | `acf(series, nlags=10)` |

### [07 · A/B Testing Fundamentals](level-2/07-ab-testing-fundamentals.md)

| Task | Code |
|---|---|
| Sample size for a proportion test | `NormalIndPower().solve_power(effect_size, alpha, power)` |
| Two-proportion z-test | `proportions_ztest([s1, s2], [n1, n2])` |
| CI on the difference | `confint_proportions_2indep(...)` |
| Effect size (Cohen's h) | `proportion_effectsize(p1, p2)` |

### [08 · SQL for Data Scientists](level-2/08-sql-for-data-scientists.md)

| Task | SQL |
|---|---|
| Filter rows | `WHERE condition` |
| Filter after aggregation | `HAVING condition` |
| Combine tables | `JOIN ... ON key` / `LEFT JOIN` |
| Per-group value, keep rows | `SUM(x) OVER (PARTITION BY k)` |
| Rank/number within group | `ROW_NUMBER() OVER (PARTITION BY k ORDER BY col)` |
| Named intermediate query | `WITH name AS (...) SELECT ...` |

### [09 · Jupyter Notebook Best Practices](level-2/09-jupyter-best-practices.md)

| Practice | Why |
|---|---|
| Restart & Run All before trusting results | Catches out-of-order execution bugs |
| One idea per cell | Isolates failures, cleaner diffs |
| Move logic to `.py` modules | Testable, reusable, reviewable |
| Pin dependencies | Reproducible environment |
| Clear outputs before commit | Small, readable git history |
| Seed randomness | Reproducible numbers |

### [10 · Project — A/B Test Analysis](level-2/10-project-ab-test-analysis.md)

| Step | Why it matters |
|---|---|
| Check covariate balance across groups | Confirms randomization worked |
| Test the pooled/primary metric first | Answers the pre-registered question |
| Break out by segment | Pooled averages can hide (or fake) effects |
| Re-test significant segments individually | Confirms the segment story statistically |
| Recommendation ties back to business action | The point of the whole analysis |

## Level 3 · Advanced

### [01 · Advanced Statistical Modeling](level-3/01-advanced-statistical-modeling.md)

| Task | Code |
|---|---|
| Multiple regression | `smf.ols("y ~ x1 + x2", data=df).fit()` |
| Check multicollinearity | `variance_inflation_factor(X, i)` |
| Shrink coefficients | `Ridge(alpha=...)`, `Lasso(alpha=...)` |
| Model counts | `smf.glm("y ~ x", data=df, family=sm.families.Poisson())` |

### [02 · Experimentation Platforms &amp; Causal Inference Basics](level-3/02-causal-inference-basics.md)

| Method | Use when |
|---|---|
| Regression adjustment | You can measure and include the key confounders |
| Propensity score matching | Many confounders; want treated/control comparability check |
| Difference-in-differences | Treatment starts at a known time for one group |
| Randomized A/B test | Still the strongest design — use it whenever you can |

### [03 · Advanced Time Series Forecasting](level-3/03-advanced-time-series-forecasting.md)

| Model | Good for |
|---|---|
| Holt-Winters | Trend + seasonality, fast, few data requirements |
| ARIMA | Non-seasonal autocorrelated series |
| SARIMA | Trend + seasonality, more tunable than Holt-Winters |
| MAE / MAPE | Compare forecast accuracy across models |
| Rolling-origin CV | Robust accuracy estimate, avoids one lucky split |

### [04 · Working with Big Data](level-3/04-working-with-big-data.md)

| Technique | Fixes |
|---|---|
| `astype("category")`, `downcast=` | Wasteful default dtypes |
| `pd.read_csv(..., chunksize=N)` | File doesn't fit in memory |
| `to_parquet` / `read_parquet(columns=...)` | Slow repeated reads, unneeded columns |
| `dask.dataframe` | Needs parallelism or exceeds single-machine memory |

### [05 · Model Interpretability](level-3/05-model-interpretability.md)

| Tool | Answers |
|---|---|
| Linear coefficients | "How much and which direction, overall?" |
| `feature_importances_` | "What does the tree split on most?" (biased) |
| `permutation_importance` | "What does performance depend on?" (honest, any model) |
| SHAP | "Why did this one prediction come out this way?" |
| Partial dependence | "What's the average shape of this feature's effect?" |

### [06 · Data Storytelling for Executives](level-3/06-data-storytelling-executives.md)

| Instinct to resist | Do instead |
|---|---|
| Lead with methodology | Lead with the conclusion and its size |
| Show every chart you made | Show the one or two that carry the argument |
| Hedge with vague language | Give a concrete range tied to a stated assumption |
| End with "more research needed" | End with a specific, actionable recommendation |

### [07 · Building Dashboards (Streamlit)](level-3/07-building-dashboards-streamlit.md)

| Task | Code |
|---|---|
| Cache expensive loads | `@st.cache_data` above a function |
| Filter widget | `st.multiselect(...)`, `st.date_input(...)` |
| KPI tile | `st.metric(label, value, delta=)` |
| Interactive chart | `st.plotly_chart(fig, use_container_width=True)` |
| Sidebar controls | `with st.sidebar: ...` |
| Run locally | `streamlit run app.py` |

### [08 · Reproducible Research Practices](level-3/08-reproducible-research.md)

| Practice | What it guards against |
|---|---|
| `raw/` vs `processed/` separation | Accidentally overwriting original data |
| Pinned `requirements.txt` + lock file | "Works on my machine" dependency drift |
| Seed every random operation | Non-reproducible numbers |
| Hash or version raw data files | Unnoticed silent data changes |
| One pipeline entry point | Manual, undocumented multi-step reruns |
| Documented limitations | Results misapplied outside their valid scope |

### [09 · Data Ethics &amp; Bias](level-3/09-data-ethics-bias.md)

| Concept | What to check |
|---|---|
| Proxy discrimination | Correlate candidate features with protected attributes, even if unused |
| Demographic parity | Equal selection rate across groups |
| Equal opportunity | Equal TPR across groups (for people who should be selected) |
| Equalized odds | Equal TPR and FPR across groups |
| Impossibility results | Parity and opportunity can't both hold exactly if base rates differ |
| Mitigation | Reweighting → feature removal → threshold adjustment → target redefinition → don't deploy |

### [10 · Project — Interactive Data Dashboard](level-3/10-project-interactive-dashboard.md)

| Piece | Purpose |
|---|---|
| `st.cache_data` | Avoid re-reading/re-parsing the source file on every interaction |
| Sidebar widgets | Filters that recompute the whole page reactively |
| `st.metric` | At-a-glance KPIs above the fold |
| `st.stop()` | Fail gracefully on empty/invalid filter states |
| `st.download_button` | Let users take the exact filtered slice with them |
| `st.expander` | Hide raw data by default without removing access to it |

## Level 4 · Master

### [01 · Leading Data Science Teams](level-4/01-leading-data-science-teams.md)

| Tool | Use it for |
|---|---|
| Team topology by size | Deciding when to split into pods / add tech leads |
| 1:1 agenda template | Keeping 1:1s from degrading into status updates |
| Review-by-questions | Scaling review without becoming the bottleneck |
| Impact × confidence / effort | Making prioritization tradeoffs explicit and defensible |
| Failure-mode table | Self-checking your own first six months as a lead |

### [02 · Advanced Causal Inference](level-4/02-advanced-causal-inference.md)

| Method | Core assumption | Main threat |
|---|---|---|
| DiD | Parallel trends absent treatment | Diverging trends for other reasons |
| IV | Relevance + exclusion restriction | Weak or invalid instrument |
| RD | Continuity of confounders across cutoff | Manipulation of running variable near cutoff |
| Synthetic control | Good pre-period fit implies good counterfactual | Poor donor pool / pre-fit |

### [03 · Building a Data Science Platform](level-4/03-data-science-platform.md)

| Component | Solves |
|---|---|
| Feature store | Training/serving skew, duplicated feature logic |
| Experiment tracking | "What have we already tried?" institutional memory |
| Model registry | Auditable promotion and rollback |
| Orchestration | Scheduled pipelines that don't depend on human memory |
| Governance layer | Who can access what data, PII handling |

### [04 · MLOps for Data Scientists](level-4/04-mlops-for-data-scientists.md)

| Stage | Data scientist's responsibility |
|---|---|
| Packaging | Save model + exact preprocessing/feature order together |
| Serving choice | Default to batch; justify real-time with a concrete need |
| Operational monitoring | Usually owned by platform/infra, but know what's tracked |
| Data drift monitoring | Often the DS's job — you know what "normal" looks like |
| Retraining trigger | Define explicitly; don't leave it to "someone will notice" |

### [05 · Advanced Experimentation at Scale](level-4/05-advanced-experimentation-scale.md)

| Problem | Cause | Fix |
|---|---|---|
| False "significant" metrics | Checking many metrics, one pre-registered primary | FDR correction (`fdr_bh`) on secondary metrics |
| Contaminated experiment reads | Overlapping experiments on shared users | Mutually exclusive layers, or explicit interaction test |
| Overstated long-run lift | Novelty/primacy effect | Look at metric trend over time, not just endpoint average |
| Inflated false positive rate | Peeking at p-values daily and stopping early | Fixed duration, or a proper sequential testing method |

### [06 · Data Science Strategy &amp; Roadmapping](level-4/06-strategy-roadmapping.md)

| Practice | Prevents |
|---|---|
| Business-question-first framing | Technique-driven roadmaps no one can prioritize against |
| Explicit dependency chain | Building downstream work before its prerequisite |
| Confidence + risk stated upward | Overpromising, and losing credibility when reality diverges |
| Resourcing tradeoff table | Reflexively adding headcount as the only lever |
| Quarterly re-check against findings | A roadmap that ignores its own results |

### [07 · Cross-Functional Stakeholder Management](level-4/07-stakeholder-management.md)

| Situation | Move |
|---|---|
| Vague request arrives | Ask what decision it feeds before starting the analysis |
| Result contradicts stakeholder's intuition | Distinguish "wrong about data" vs. "disagree on action" |
| Result is statistically uncertain | State the interval and its decision implication explicitly |
| Readout to a time-pressed exec | Decision → answer → confidence → action, methods in appendix |
| Feeling pressure to soften a result | Name the instinct; present the finding transparently anyway |

### [08 · Ethics &amp; Governance at Scale](level-4/08-ethics-governance-scale.md)

| Practice | Prevents |
|---|---|
| Risk tiering rubric | Review effort misallocated relative to actual stakes |
| Mandatory model card (high tier) | Undocumented intended use, gaps, and evaluation |
| Subgroup performance reporting | Aggregate metrics hiding a poorly-served subgroup |
| Launch gate (not a suggestion) | Review being cut under deadline pressure |
| Re-audit on retrain | Bias silently reintroduced by a retraining cycle |
| Re-review on repurposing | A model's fairness evaluation not transferring to a new use case |

### [09 · Career Growth: IC to Principal Data Scientist](level-4/09-career-growth.md)

| Level | Differentiator |
|---|---|
| Junior → Mid | Independence on well-scoped work |
| Mid → Senior | Owns ambiguous problems, decides what to work on |
| Senior → Staff | Cross-team influence without formal authority; org-level outcomes |
| Staff → Principal | Sets direction across the org, judged on org-wide outcomes |
| Any level | Multiplier effect (mentorship, reusable frameworks) compounds faster than solo output |

### [10 · Capstone — Full Data Science Project from Question to Recommendation](level-4/10-capstone-full-project.md)

| Stage | Module(s) it draws on |
|---|---|
| Narrowing a vague ask into a decision-linked question | L4 Module 07 |
| Exploratory analysis and descriptive segmentation | L1-L2 EDA modules |
| Feature engineering and a classification pipeline | L2-L3 modeling modules |
| Coefficient/feature-importance interpretation | L3 model interpretation modules |
| Correlation-vs-causation discipline | L4 Module 02 |
| Risk-scoring deployment vs. experiment-before-reallocation | L4 Modules 04-05 |
| Decision-focused readout structure | L4 Module 07 |

---

Go deeper with the full [Data Science Mastery Path](index.md) curriculum or test yourself with the [interview questions](interview-questions.md).
