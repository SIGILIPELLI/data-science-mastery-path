---
title: "Data Science Interview Questions and Answers (179 Questions)"
description: "179 Data Science interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the Data Science Mastery Path."
---

# Data Science Interview Questions and Answers

**179 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free Data Science Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Entry

### [01 · What Is Data Science?](level-1/01-what-is-data-science.md)

**Q1.** What does data science combine?

**Answer:** Statistics, programming and domain knowledge to turn data into decisions

Each leg is needed.

**Q2.** Which question best starts a data science project?

**Answer:** A specific business question the analysis should answer

Start from the decision.

**Q3.** What's the typical workflow order?

**Answer:** Question, data collection, cleaning, exploration, modeling/analysis, communication

Iterate as you learn.

**Q4.** Why is domain knowledge important?

**Answer:** It tells you what the numbers mean and which results are plausible

Context prevents silly conclusions.

### [02 · Python for Data Science](level-1/02-python-for-data-science.md)

**Q5.** What does this print?

```python
import statistics as s
d=[2,4,4,4,5,5,7,9]
print(s.mean(d), s.median(d), s.mode(d), round(s.pstdev(d),2))
```

**Answer:** 5 4.5 4 2.0

Mean 5, median 4.5, mode 4, population SD 2. Verified by running it.

**Q6.** Why use NumPy arrays over Python lists for numeric work?

**Answer:** Fixed-type contiguous arrays enable fast vectorized operations

Avoid Python loops.

**Q7.** What is a pandas DataFrame?

**Answer:** A labeled, spreadsheet-like table built on NumPy

Rows and named columns.

**Q8.** What's vectorization?

**Answer:** Applying operations to whole arrays at once instead of looping in Python

Faster and cleaner.

### [03 · Data Cleaning &amp; Wrangling](level-1/03-data-cleaning-wrangling.md)

**Q9.** What does this print?

```python
rows=[('A',1),('a ',1),('B',2),('A',1)]
clean={(n.strip().upper(),v) for n,v in rows}
print(sorted(clean))
```

**Answer:** [('A', 1), ('B', 2)]

Normalize text, then deduplicate. Verified by running it.

**Q10.** What are common ways to handle missing values?

**Answer:** Drop, impute (mean/median/model) or flag, depending on why they're missing

Understand the mechanism.

**Q11.** Why standardize text case and whitespace?

**Answer:** Otherwise 'USA' and 'usa ' count as different categories

Normalize first.

**Q12.** Why keep the raw data untouched?

**Answer:** So cleaning is reproducible and reversible

Script every step.

### [04 · Exploratory Data Analysis (EDA)](level-1/04-exploratory-data-analysis.md)

**Q13.** What's the point of EDA?

**Answer:** Understand shape, distributions, outliers and relationships before answering a question

Skipping it invites errors.

**Q14.** A column's mean is much higher than its median. What does that suggest?

**Answer:** Right skew or outliers pulling the mean

Look at the distribution.

**Q15.** What does `df.describe()` summarize?

**Answer:** Count, mean, spread and quartiles of numeric columns

First look.

**Q16.** Why check for duplicates and impossible values early?

**Answer:** They distort every later statistic

Sanity checks.

### [05 · Data Visualization Basics](level-1/05-data-visualization-basics.md)

**Q17.** Which chart shows a distribution of one numeric variable?

**Answer:** Histogram

Bin choice matters.

**Q18.** Which chart shows the relationship between two numeric variables?

**Answer:** Scatter plot

Look for trend and outliers.

**Q19.** Why should bar charts start at zero?

**Answer:** Truncated axes exaggerate differences

Bar length encodes value.

**Q20.** What makes a chart honest and clear?

**Answer:** Labeled axes, units, a title that states the takeaway and minimal clutter

Design for the reader.

### [06 · Statistics Fundamentals](level-1/06-statistics-fundamentals.md)

**Q21.** What does this print?

```python
import math
d=[10,12,9,11,13]
m=sum(d)/len(d); sd=(sum((x-m)**2 for x in d)/(len(d)-1))**.5
print(m, round(sd,3), round(sd/math.sqrt(len(d)),3))
```

**Answer:** 11.0 1.581 0.707

Sample SD uses n-1; standard error is sd/sqrt(n). Verified by running it.

**Q22.** What does a p-value of 0.03 mean?

**Answer:** If the null hypothesis were true, data this extreme would occur about 3% of the time

It isn't the probability the hypothesis is true.

**Q23.** What does a 95% confidence interval mean?

**Answer:** The procedure captures the true value in about 95% of repeated samples

Common misinterpretation.

**Q24.** Why does a larger sample narrow the CI?

**Answer:** Standard error shrinks as 1/sqrt(n)

More data, more precision.

### [07 · Correlation &amp; Basic Regression](level-1/07-correlation-basic-regression.md)

**Q25.** What does this print?

```python
x=[1,2,3,4,5]; y=[2,4,5,4,5]
n=5; mx=sum(x)/n; my=sum(y)/n
b=sum((a-mx)*(c-my) for a,c in zip(x,y))/sum((a-mx)**2 for a in x)
print(round(b,2), round(my-b*mx,2))
```

**Answer:** 0.6 2.2

Least squares slope and intercept. Verified by running it.

**Q26.** Does correlation imply causation?

**Answer:** No: confounders, reverse causality and chance can create correlation

Think about the mechanism.

**Q27.** What does R^2 measure?

**Answer:** The share of variance in y explained by the model

Not proof of correctness.

**Q28.** What does an outlier do to a regression line?

**Answer:** It can pull the line substantially (high leverage)

Inspect residuals.

### [08 · Working with Real-World Messy Datasets](level-1/08-messy-datasets.md)

**Q29.** An order table has 'USA', 'U.S.A.' and 'United States'. What's the fix?

**Answer:** Map variants to one canonical value

Normalization.

**Q30.** What should you do about missing customer ids?

**Answer:** Understand why, then decide to impute, flag or exclude, and document it

Be transparent.

**Q31.** Why log every cleaning decision?

**Answer:** So the analysis can be reviewed and reproduced

Reproducibility.

**Q32.** What's a sanity check after cleaning?

**Answer:** Row counts and totals reconcile with the source

Verify.

### [09 · Communicating Findings](level-1/09-communicating-findings.md)

**Q33.** What should lead a report to executives?

**Answer:** The finding and recommendation, then the supporting evidence

Answer first.

**Q34.** What makes a chart title effective?

**Answer:** It states the takeaway ('Churn doubled after the price change')

Tell the story.

**Q35.** Why state uncertainty and limitations?

**Answer:** Honest framing builds trust and prevents misuse

Don't overclaim.

**Q36.** What's a good call to action?

**Answer:** A specific recommended next step with owner and expected impact

Enable decisions.

### [10 · Project — End-to-End Data Analysis Report](level-1/10-capstone-project.md)

**Q37.** What's the goal of the churn analysis?

**Answer:** Understand why customers leave and recommend actions

Business question first.

**Q38.** Which comparison helps explain churn?

**Answer:** Churn rate by segment, tenure and engagement

Compare groups.

**Q39.** Why report effect sizes with significance?

**Answer:** A significant result can still be tiny and unimportant

Practical significance.

**Q40.** What ends the capstone?

**Answer:** A clear recommendation backed by evidence and caveats

Decision-ready.

## Level 2 · Intermediate

### [01 · Advanced pandas (groupby, merge, pivot)](level-2/01-advanced-pandas.md)

**Q41.** What does this groupby return?

```sql
create table s(r text, v int); insert into s values('n',10),('n',20),('s',5);

select r, avg(v), count(*) from s group by r order by r;
```

**Answer:** n,15.0,2; s,5.0,1

Aggregate within groups (same idea as pandas groupby). Verified by running it in SQLite (rows shown as comma-separated, separated by semicolons).

**Q42.** What's the risk of a many-to-many merge?

**Answer:** Row explosion duplicating records and inflating totals

Check key uniqueness.

**Q43.** Why prefer vectorized pandas ops to .apply with a Python function?

**Answer:** They're much faster

Avoid row loops.

**Q44.** What does `pd.merge(how='left')` keep?

**Answer:** All rows from the left table with matches where present

Else NaN.

### [02 · Statistical Testing Deep Dive](level-2/02-statistical-testing.md)

**Q45.** Which test compares the means of two independent groups?

**Answer:** Two-sample t-test (or Mann-Whitney if non-normal)

Check assumptions.

**Q46.** Which test is for two categorical variables?

**Answer:** Chi-square test of independence

Contingency table.

**Q47.** Which handles more than two group means?

**Answer:** ANOVA (with post-hoc tests)

Multiple comparisons inflate false positives.

**Q48.** What is the multiple comparisons problem?

**Answer:** Running many tests raises the chance of false positives; correct for it

Bonferroni, FDR.

### [03 · Advanced Visualization (plotly)](level-2/03-advanced-visualization.md)

**Q49.** What does interactive Plotly add over static charts?

**Answer:** Hover, zoom and toggling series for exploration

Good for exploring.

**Q50.** When use small multiples (facets)?

**Answer:** To compare the same chart across categories

Easy comparison.

**Q51.** What's a heatmap good for?

**Answer:** Showing magnitudes across two categorical dimensions (e.g. correlation matrix)

Use sensible color scales.

**Q52.** Why avoid rainbow color scales?

**Answer:** They mislead perception of magnitude and hurt accessibility

Use perceptually uniform maps.

### [04 · Feature Engineering Basics](level-2/04-feature-engineering.md)

**Q53.** What does one-hot encoding do?

**Answer:** Turns a categorical variable into binary columns per category

Avoids false ordering.

**Q54.** Why scale features for some models?

**Answer:** Distance/gradient-based models are sensitive to feature magnitude

Standardize.

**Q55.** What is data leakage in feature engineering?

**Answer:** Using information unavailable at prediction time (e.g. future data)

Split before fitting transforms.

**Q56.** Why create interaction or date-part features?

**Answer:** Models can't infer them automatically from raw columns

Domain knowledge.

### [05 · Intro to Machine Learning for Data Science](level-2/05-intro-machine-learning.md)

**Q57.** What's a baseline model for?

**Answer:** A simple benchmark that any real model must beat

Majority class or mean.

**Q58.** Why split train and test data?

**Answer:** To estimate performance on unseen data

Avoid overfitting.

**Q59.** What is overfitting?

**Answer:** Model memorizes training noise and performs worse on new data

Regularize, simplify.

**Q60.** Why is accuracy misleading on imbalanced data?

**Answer:** Predicting the majority class gets high accuracy but misses the minority

Use precision/recall/AUC.

### [06 · Time Series Analysis Basics](level-2/06-time-series-basics.md)

**Q61.** Why can't you randomly shuffle time series for validation?

**Answer:** It leaks the future into training; use time-based splits

Respect order.

**Q62.** What are trend, seasonality and noise?

**Answer:** Long-term direction, repeating patterns and random variation

Decomposition.

**Q63.** What does a rolling mean do?

**Answer:** Smooths noise by averaging a moving window

Window size matters.

**Q64.** What does this print?

```python
x=[1,2,3,4,5,6]
print([round(sum(x[i:i+3])/3,1) for i in range(len(x)-2)])
```

**Answer:** [2.0, 3.0, 4.0, 5.0]

3-point rolling mean. Verified by running it.

### [07 · A/B Testing Fundamentals](level-2/07-ab-testing-fundamentals.md)

**Q65.** What's the purpose of randomization in A/B tests?

**Answer:** To balance known and unknown confounders across groups

Enables causal claims.

**Q66.** Why decide the sample size and metric before the test?

**Answer:** To avoid peeking and p-hacking

Pre-registration.

**Q67.** What's wrong with stopping the test as soon as p&lt;0.05?

**Answer:** Repeated peeking inflates false positives

Fix the horizon or use sequential methods.

**Q68.** What does statistical power mean?

**Answer:** The probability of detecting a real effect of a given size

Plan sample size.

### [08 · SQL for Data Scientists](level-2/08-sql-for-data-scientists.md)

**Q69.** What does this return?

```sql
create table u(id int, plan text); insert into u values(1,'a'),(2,'b'),(3,'a'); create table e(uid int, n int); insert into e values(1,5),(1,3),(3,2);

select u.plan, sum(coalesce(e.n,0)) from u left join e on e.uid=u.id group by u.plan order by u.plan;
```

**Answer:** a,10; b,0

LEFT JOIN keeps users without events. Verified by running it in SQLite (rows shown as comma-separated, separated by semicolons).

**Q70.** Why pull and aggregate data in SQL instead of pandas?

**Answer:** Databases aggregate large data efficiently near the source

Push computation down.

**Q71.** What does `GROUP BY` plus `HAVING` do?

**Answer:** Filters groups after aggregation

HAVING count(*)&gt;5.

**Q72.** Why use parameterized queries?

**Answer:** To prevent SQL injection and quoting bugs

Bind values.

### [09 · Jupyter Notebook Best Practices](level-2/09-jupyter-best-practices.md)

**Q73.** What's the main risk of notebooks?

**Answer:** Hidden state from out-of-order cell execution

Restart and run all.

**Q74.** How do you keep a notebook reproducible?

**Answer:** Run top to bottom, pin dependencies and set seeds

Rerun from scratch.

**Q75.** Where should reusable code go?

**Answer:** In modules imported by notebooks

Test it.

**Q76.** Why strip outputs before committing?

**Answer:** Cleaner diffs and no accidental data leakage

nbstripout.

### [10 · Project — A/B Test Analysis](level-2/10-project-ab-test-analysis.md)

**Q77.** What do you check first in an experiment analysis?

**Answer:** Sample ratio mismatch and data quality

Invalid assignment invalidates results.

**Q78.** What metrics should be reported?

**Answer:** Effect size with confidence interval, not only p-value

Practical significance.

**Q79.** What does a guardrail metric protect?

**Answer:** Against harming something important like latency or revenue

Check side effects.

**Q80.** What's the final deliverable?

**Answer:** A recommendation (ship, iterate, stop) with evidence and caveats

Decision-ready.

## Level 3 · Advanced

### [01 · Advanced Statistical Modeling](level-3/01-advanced-statistical-modeling.md)

**Q81.** What does logistic regression model?

**Answer:** The probability of a binary outcome via log-odds

Coefficients are log-odds.

**Q82.** What does regularization (L1/L2) do?

**Answer:** Penalizes large coefficients to reduce overfitting

L1 can zero out features.

**Q83.** What's multicollinearity?

**Answer:** Highly correlated predictors that make coefficients unstable

Check VIF.

**Q84.** Why check residuals?

**Answer:** To verify model assumptions like linearity and constant variance

Diagnostics.

### [02 · Experimentation Platforms &amp; Causal Inference Basics](level-3/02-causal-inference-basics.md)

**Q85.** Why is observational correlation not enough for causal claims?

**Answer:** Confounders can drive both variables

Randomization or adjustment needed.

**Q86.** What's a confounder?

**Answer:** A variable influencing both treatment and outcome

Adjust for it.

**Q87.** What does difference-in-differences estimate?

**Answer:** The effect of a treatment by comparing changes over time between treated and control groups

Assumes parallel trends.

**Q88.** What's the gold standard for causal effects?

**Answer:** A randomized controlled experiment

If feasible and ethical.

### [03 · Advanced Time Series Forecasting](level-3/03-advanced-time-series-forecasting.md)

**Q89.** What does ARIMA model?

**Answer:** A series via autoregression, differencing and moving average terms

Stationarity required.

**Q90.** What is stationarity?

**Answer:** Statistical properties like mean and variance don't change over time

Differencing helps.

**Q91.** How do you evaluate forecasts?

**Answer:** On a held-out future window using metrics like MAE/MAPE

Walk-forward validation.

**Q92.** Why compare to a naive forecast?

**Answer:** Complex models must beat 'same as last period' to be useful

Baselines.

### [04 · Working with Big Data](level-3/04-working-with-big-data.md)

**Q93.** What's the first approach when data doesn't fit in memory?

**Answer:** Sample, use chunked/streaming processing or a columnar engine like DuckDB/Parquet

Avoid unnecessary scale.

**Q94.** Why prefer Parquet over CSV for large data?

**Answer:** Columnar compression and column pruning cut I/O

Typed and fast.

**Q95.** When do you need Spark or Dask?

**Answer:** When single-machine tools can't handle volume or compute time

Distributed.

**Q96.** Why push filters and aggregations early?

**Answer:** Reduce data volume before expensive steps

Predicate pushdown.

### [05 · Model Interpretability](level-3/05-model-interpretability.md)

**Q97.** What do SHAP values explain?

**Answer:** How much each feature contributed to a specific prediction

Local and global views.

**Q98.** Why interpretability matters?

**Answer:** Trust, debugging, fairness and regulatory needs

Understand failures.

**Q99.** What does permutation importance do?

**Answer:** Measures how much shuffling a feature hurts performance

Model-agnostic.

**Q100.** Caution when reading feature importance?

**Answer:** Correlated features share importance and importance isn't causation

Interpret carefully.

### [06 · Data Storytelling for Executives](level-3/06-data-storytelling-executives.md)

**Q101.** What do executives need first?

**Answer:** The answer and the decision it implies

Lead with the point.

**Q102.** Why limit slides to one message each?

**Answer:** Clarity and retention

One chart, one takeaway.

**Q103.** How do you handle uncertainty with executives?

**Answer:** State it plainly with ranges and implications

Honesty builds trust.

**Q104.** What's a good closing?

**Answer:** A recommendation with next steps and owners

Drive action.

### [07 · Building Dashboards (Streamlit)](level-3/07-building-dashboards-streamlit.md)

**Q105.** What does Streamlit do?

**Answer:** Turns Python scripts into interactive web apps with widgets

Reruns the script on interaction.

**Q106.** What does @st.cache_data help with?

**Answer:** Avoiding recomputation of expensive data loading

Faster apps.

**Q107.** Why keep business logic out of UI code?

**Answer:** Testability and reuse

Separate concerns.

**Q108.** What should a dashboard answer?

**Answer:** A defined set of questions for a specific audience

Avoid clutter.

### [08 · Reproducible Research Practices](level-3/08-reproducible-research.md)

**Q109.** What makes an analysis reproducible?

**Answer:** Versioned code and data, pinned environment and fixed seeds

Others can rerun it.

**Q110.** Why pin dependency versions?

**Answer:** Library updates can change results

requirements.lock.

**Q111.** Why use Git for analysis?

**Answer:** History, review and collaboration

Track changes.

**Q112.** What should be documented?

**Answer:** Data sources, assumptions and how to run

README.

### [09 · Data Ethics &amp; Bias](level-3/09-data-ethics-bias.md)

**Q113.** Where can bias enter a model?

**Answer:** Data collection, labels, features, evaluation and deployment

Whole lifecycle.

**Q114.** Why evaluate performance by subgroup?

**Answer:** Averages can hide large gaps for some groups

Disaggregate.

**Q115.** What is proxy discrimination?

**Answer:** Using a feature correlated with a protected attribute to the same effect

Zip code as race proxy.

**Q116.** What should you do upon finding bias?

**Answer:** Document, mitigate, monitor and decide if the use is appropriate

Responsible practice.

### [10 · Project — Interactive Data Dashboard](level-3/10-project-interactive-dashboard.md)

**Q117.** What's the first step of the dashboard project?

**Answer:** Define the users and the questions it must answer

Audience first.

**Q118.** Why add filters carefully?

**Answer:** Too many options confuse; offer those that matter

Design for tasks.

**Q119.** How do you verify numbers in the dashboard?

**Answer:** Reconcile against source queries

Trust matters.

**Q120.** What should you do about slow queries?

**Answer:** Cache or pre-aggregate

Performance.

## Level 4 · Master

### [01 · Leading Data Science Teams](level-4/01-leading-data-science-teams.md)

**Q121.** What does a data science lead do beyond analysis?

**Answer:** Prioritizes problems, grows people and connects work to business outcomes

Leadership.

**Q122.** How should work be prioritized?

**Answer:** By expected business value and feasibility

Impact vs effort.

**Q123.** How do you grow team members?

**Answer:** Stretch assignments, feedback and review culture

Mentoring.

**Q124.** What's a healthy review practice?

**Answer:** Code and analysis review for correctness and clarity

Quality.

### [02 · Advanced Causal Inference](level-4/02-advanced-causal-inference.md)

**Q125.** What does an instrumental variable need?

**Answer:** Affects treatment, affects outcome only through treatment, and is independent of confounders

Hard to find valid ones.

**Q126.** What does propensity score matching do?

**Answer:** Balances observed covariates between treated and untreated units

Unobserved confounding remains.

**Q127.** What's regression discontinuity?

**Answer:** Comparing units just above and below a threshold that determines treatment

Local effect.

**Q128.** What's the limitation of all observational methods?

**Answer:** They depend on untestable assumptions

Be transparent.

### [03 · Building a Data Science Platform](level-4/03-data-science-platform.md)

**Q129.** What does a data science platform provide?

**Answer:** Shared compute, data access, experiment tracking and deployment tooling

Productivity.

**Q130.** What does experiment tracking record?

**Answer:** Parameters, code version, data version and metrics

Reproducibility.

**Q131.** Why standardize environments?

**Answer:** Reproducibility across people and machines

Containers.

**Q132.** What does self-service enable?

**Answer:** Scientists work without waiting on infrastructure teams

Guardrails.

### [04 · MLOps for Data Scientists](level-4/04-mlops-for-data-scientists.md)

**Q133.** What does MLOps add to modeling?

**Answer:** Versioning, deployment, monitoring and retraining processes

Operate models.

**Q134.** What's model drift?

**Answer:** Performance degrades as data distribution or relationships change

Monitor and retrain.

**Q135.** Why version data and models?

**Answer:** To reproduce and roll back

Registry.

**Q136.** What should a deployment include?

**Answer:** Tests, monitoring and a rollback plan

Safe release.

### [05 · Advanced Experimentation at Scale](level-4/05-advanced-experimentation-scale.md)

**Q137.** What's CUPED?

**Answer:** Variance reduction using pre-experiment covariates to detect smaller effects

Increases power.

**Q138.** What's a network effect problem in A/B tests?

**Answer:** Treated users influence control users, violating independence

Cluster randomization.

**Q139.** Why run many experiments with a shared platform?

**Answer:** Consistency, guardrails and learning at scale

Experimentation culture.

**Q140.** What risk does running many metrics create?

**Answer:** False positives; predefine primary metrics and correct

Discipline.

### [06 · Data Science Strategy &amp; Roadmapping](level-4/06-strategy-roadmapping.md)

**Q141.** What does a data strategy align?

**Answer:** Data and analytics investments with business goals

Value-driven.

**Q142.** How do you prioritize roadmap items?

**Answer:** Impact, effort, risk and dependencies

Transparent criteria.

**Q143.** Why include quick wins and long bets?

**Answer:** Build credibility while investing in capability

Balanced portfolio.

**Q144.** How do you track progress?

**Answer:** Outcome metrics tied to business results

Measure value.

### [07 · Cross-Functional Stakeholder Management](level-4/07-stakeholder-management.md)

**Q145.** How do you handle a stakeholder demanding a result you can't support?

**Answer:** Explain evidence and limits, and offer alternatives

Integrity.

**Q146.** What clarifies expectations early?

**Answer:** Agreeing the question, success criteria and timeline

Scope.

**Q147.** Why communicate regularly?

**Answer:** Prevents surprises and builds trust

Cadence.

**Q148.** What's a good way to deliver bad news?

**Answer:** Early, with facts, impact and a proposed path

Be direct.

### [08 · Ethics &amp; Governance at Scale](level-4/08-ethics-governance-scale.md)

**Q149.** What does AI governance include?

**Answer:** Policies, risk review, documentation and accountability

Scale with risk.

**Q150.** What is a model card?

**Answer:** Documentation of a model's purpose, data, performance and limits

Transparency.

**Q151.** Why do impact assessments matter?

**Answer:** Identify harms before deployment

Proactive.

**Q152.** Who is accountable for a model's outcomes?

**Answer:** A named owner

Assign ownership.

### [09 · Career Growth: IC to Principal Data Scientist](level-4/09-career-growth.md)

**Q153.** What distinguishes senior data scientists?

**Answer:** Problem framing, judgment and influence beyond technical skill

Scope grows.

**Q154.** How do you build credibility?

**Answer:** Deliver reliable, communicated results and own mistakes

Track record.

**Q155.** What skills compound?

**Answer:** Statistics, communication and domain expertise

Durable.

**Q156.** How do you choose a specialization?

**Answer:** Align interests, strengths and market needs

Balance.

### [10 · Capstone — Full Data Science Project from Question to Recommendation](level-4/10-capstone-full-project.md)

**Q157.** What does the capstone show?

**Answer:** An end-to-end project from question to communicated, deployed recommendation

Full lifecycle.

**Q158.** How do you demonstrate rigor?

**Answer:** Validation, sensitivity checks and clear limitations

Evidence.

**Q159.** What would make it production-ready?

**Answer:** Reproducibility, monitoring and documentation

Operate it.

**Q160.** What's the final measure of success?

**Answer:** Business impact from decisions informed by the work

Outcome.

## Mixed questions across all levels

**Q161.** A manager asks for 'the data science'. What should you do first?

**Answer:** Clarify the business question and the decision the analysis will inform

Start from the decision.

**Q162.** Customer names appear as 'USA', 'U.S.A.' and 'United States'. What's the correct cleaning step?

**Answer:** Map variants to one canonical value and log the rule

Normalize and document.

**Q163.** What does this print?

```python
import statistics as s
d=[1,2,2,3,100]
print(s.mean(d), s.median(d))
```

**Answer:** 21.6 2

The outlier drags the mean but not the median. Verified by running it.

**Q164.** What does this print for a sample SD?

```python
import statistics as s
print(round(s.stdev([10,12,9,11,13]),3), round(s.pstdev([10,12,9,11,13]),3))
```

**Answer:** 1.581 1.414

Sample SD divides by n-1. Verified by running it.

**Q165.** A study finds p=0.03. Which statement is correct?

**Answer:** If there were no effect, data this extreme would occur about 3% of the time

p-values aren't the probability a hypothesis is true.

**Q166.** Ice cream sales and drownings are correlated. What's the best explanation?

**Answer:** A confounder (hot weather) drives both

Correlation isn't causation.

**Q167.** A model gets 95% accuracy predicting churn when 5% churn. What do you check?

**Answer:** Whether it just predicts 'no churn'; look at precision, recall and a baseline

Imbalanced classes.

**Q168.** Features were standardized using the whole dataset before the train/test split. What went wrong?

**Answer:** Data leakage: test information influenced the training transform

Fit transforms on training data only.

**Q169.** You shuffle a daily sales series randomly before splitting. What's the problem?

**Answer:** The model trains on the future to predict the past; use a time-based split

Respect time order.

**Q170.** An A/B test is stopped as soon as p dips below 0.05. Why is that risky?

**Answer:** Repeated peeking inflates false positives; fix the horizon or use sequential methods

Pre-register stopping rules.

**Q171.** The control and treatment groups are 55%/45% when 50/50 was intended. What do you suspect?

**Answer:** A sample ratio mismatch: assignment or logging is broken

Fix before analysis.

**Q172.** An A/B test shows +0.2% lift, p=0.001 on 10 million users. How do you report it?

**Answer:** Statistically significant but tiny; judge practical significance and cost

Effect size matters.

**Q173.** You ran 20 metric tests and one is p=0.04. What's the concern?

**Answer:** Multiple comparisons: expect about one false positive; correct or predefine primary metrics

False discovery.

**Q174.** Observational data show treated users do better, but they self-selected. What do you use?

**Answer:** Causal methods (matching, diff-in-diff) with stated assumptions, ideally a randomized test

Selection bias.

**Q175.** A model explains predictions with SHAP. What can you say?

**Answer:** Which features contributed to a prediction, not that they cause the outcome

Interpretability isn't causality.

**Q176.** A dashboard loads slowly on 50 million rows. What helps?

**Answer:** Pre-aggregate or cache and query only what's needed

Performance.

**Q177.** Your notebook gives different results when run top to bottom. Likely cause?

**Answer:** Hidden state from out-of-order cells; restart and run all with fixed seeds

Reproducibility.

**Q178.** A hiring model's accuracy is 90% overall but 70% for one group. What next?

**Answer:** Investigate and mitigate the disparity before deployment

Evaluate by subgroup.

**Q179.** An executive asks for a one-slide summary. What leads it?

**Answer:** The recommendation and the key evidence

Answer first.

## Practical coding questions

### A/B test z-test

Implement two_proportion_z(conv_a, n_a, conv_b, n_b) using the pooled-proportion z-test and a two-sided p-value from the normal CDF (math.erf). Analyze 200/2000 vs 250/2000 and then a small 200/2000 vs 210/2000 difference.

Starter code:

```
import math

def two_proportion_z(conv_a, n_a, conv_b, n_b):
    # return (p_a, p_b, z, two_sided_p) using the pooled-proportion z-test;
    # p-value from the normal CDF: 0.5*(1+erf(z/sqrt(2)))
    return 0, 0, 0, 1

p_a, p_b, z, p = two_proportion_z(200, 2000, 250, 2000)
print("A=%.3f B=%.3f lift=%.1f%%" % (p_a, p_b, (p_b - p_a) / p_a * 100))
print("z=%.2f p=%.4f" % (z, p))
print("significant at 5%:", p < 0.05)
_, _, z2, p2 = two_proportion_z(200, 2000, 210, 2000)
print("small difference p=%.2f significant: %s" % (p2, p2 < 0.05))
```

Expected output:

```text
A=0.100 B=0.125 lift=25.0%
z=2.50 p=0.0124
significant at 5%: True
small difference p=0.60 significant: False
```

??? success "Show a solution"

    ```
    import math

    def two_proportion_z(conv_a, n_a, conv_b, n_b):
        p_a, p_b = conv_a / n_a, conv_b / n_b
        pooled = (conv_a + conv_b) / (n_a + n_b)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
        z = (p_b - p_a) / se
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return p_a, p_b, z, p_value

    p_a, p_b, z, p = two_proportion_z(200, 2000, 250, 2000)
    print("A=%.3f B=%.3f lift=%.1f%%" % (p_a, p_b, (p_b - p_a) / p_a * 100))
    print("z=%.2f p=%.4f" % (z, p))
    print("significant at 5%:", p < 0.05)
    _, _, z2, p2 = two_proportion_z(200, 2000, 210, 2000)
    print("small difference p=%.2f significant: %s" % (p2, p2 < 0.05))
    ```

    A 2.5-point lift on 2,000 users per arm is unlikely under no difference, while a 0.5-point lift isn't distinguishable from noise. Report effect size and interval too. Verified by running the solution.

### Least squares and a warning

Implement fit_line(x, y) returning slope, intercept and R-squared by ordinary least squares. Fit a clean linear dataset, predict x=7, and then fit data with one outlier to see how it distorts the line.

Starter code:

```
def fit_line(x, y):
    # ordinary least squares: return (slope, intercept, r_squared)
    return 0, 0, 0

x = [1, 2, 3, 4, 5, 6]
y = [2.1, 4.3, 5.9, 8.2, 9.9, 12.1]
m, c, r2 = fit_line(x, y)
print("slope=%.3f intercept=%.3f r2=%.4f" % (m, c, r2))
print("prediction for x=7: %.2f" % (c + m * 7))

# Anscombe-style warning: same fit, different shape
x2 = [1, 2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 10, 30]
m2, c2, r22 = fit_line(x2, y2)
print("outlier fit slope=%.2f r2=%.2f" % (m2, r22))
```

Expected output:

```text
slope=1.974 intercept=0.173 r2=0.9983
prediction for x=7: 13.99
outlier fit slope=4.57 r2=0.70
```

??? success "Show a solution"

    ```
    def fit_line(x, y):
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        sxx = sum((a - mx) ** 2 for a in x)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        slope = sxy / sxx
        intercept = my - slope * mx
        ss_res = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(x, y))
        ss_tot = sum((b - my) ** 2 for b in y)
        return slope, intercept, 1 - ss_res / ss_tot

    x = [1, 2, 3, 4, 5, 6]
    y = [2.1, 4.3, 5.9, 8.2, 9.9, 12.1]
    m, c, r2 = fit_line(x, y)
    print("slope=%.3f intercept=%.3f r2=%.4f" % (m, c, r2))
    print("prediction for x=7: %.2f" % (c + m * 7))

    # Anscombe-style warning: same fit, different shape
    x2 = [1, 2, 3, 4, 5, 6]
    y2 = [2, 4, 6, 8, 10, 30]
    m2, c2, r22 = fit_line(x2, y2)
    print("outlier fit slope=%.2f r2=%.2f" % (m2, r22))
    ```

    The first fit is nearly perfect, while one outlier at x=6 changes the slope sharply and drops R-squared. Always plot your data and inspect residuals. Verified by running the solution.

---

More free practice: the [Data Science Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).
