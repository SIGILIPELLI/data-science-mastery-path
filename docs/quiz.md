---
title: "Final Quiz"
description: "Test what you have learned across all four levels of Data Science Mastery Path with a 19-question final quiz."
---

# Final Quiz

19 questions drawn from all four levels of this course. You need **70% (14 out of 19)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "A manager asks for 'the data science'. What should you do first?",
    "options": [
      "Start cleaning",
      "Clarify the business question and the decision the analysis will inform",
      "Pick a model",
      "Build a dashboard"
    ],
    "correctIndex": 1,
    "explanation": "Start from the decision."
  },
  {
    "question": "Customer names appear as 'USA', 'U.S.A.' and 'United States'. What's the correct cleaning step?",
    "options": [
      "Map variants to one canonical value and log the rule",
      "Leave them",
      "Drop them",
      "Pick one row"
    ],
    "correctIndex": 0,
    "explanation": "Normalize and document."
  },
  {
    "question": "What does this print?\n```python\nimport statistics as s\nd=[1,2,2,3,100]\nprint(s.mean(d), s.median(d))\n```\n",
    "options": [
      "21.6 21.6",
      "2 21.6",
      "2 2",
      "21.6 2"
    ],
    "correctIndex": 3,
    "explanation": "The outlier drags the mean but not the median. Verified by running it."
  },
  {
    "question": "What does this print for a sample SD?\n```python\nimport statistics as s\nprint(round(s.stdev([10,12,9,11,13]),3), round(s.pstdev([10,12,9,11,13]),3))\n```\n",
    "options": [
      "1.414 1.581",
      "1.581 1.581",
      "1.414 1.414",
      "1.581 1.414"
    ],
    "correctIndex": 3,
    "explanation": "Sample SD divides by n-1. Verified by running it."
  },
  {
    "question": "A study finds p=0.03. Which statement is correct?",
    "options": [
      "The effect is large",
      "There's a 3% chance the null is true",
      "There's a 97% chance the effect is real",
      "If there were no effect, data this extreme would occur about 3% of the time"
    ],
    "correctIndex": 3,
    "explanation": "p-values aren't the probability a hypothesis is true."
  },
  {
    "question": "Ice cream sales and drownings are correlated. What's the best explanation?",
    "options": [
      "Coincidence only",
      "A confounder (hot weather) drives both",
      "Drowning causes sales",
      "Ice cream causes drowning"
    ],
    "correctIndex": 1,
    "explanation": "Correlation isn't causation."
  },
  {
    "question": "A model gets 95% accuracy predicting churn when 5% churn. What do you check?",
    "options": [
      "Whether it just predicts 'no churn'; look at precision, recall and a baseline",
      "Accuracy only",
      "Ship it",
      "More epochs"
    ],
    "correctIndex": 0,
    "explanation": "Imbalanced classes."
  },
  {
    "question": "Features were standardized using the whole dataset before the train/test split. What went wrong?",
    "options": [
      "Nothing",
      "Underfitting",
      "Overfitting only",
      "Data leakage: test information influenced the training transform"
    ],
    "correctIndex": 3,
    "explanation": "Fit transforms on training data only."
  },
  {
    "question": "You shuffle a daily sales series randomly before splitting. What's the problem?",
    "options": [
      "Bias only",
      "The model trains on the future to predict the past; use a time-based split",
      "Speed",
      "Nothing"
    ],
    "correctIndex": 1,
    "explanation": "Respect time order."
  },
  {
    "question": "An A/B test is stopped as soon as p dips below 0.05. Why is that risky?",
    "options": [
      "It's fine",
      "Repeated peeking inflates false positives; fix the horizon or use sequential methods",
      "Speeds up",
      "Lowers power"
    ],
    "correctIndex": 1,
    "explanation": "Pre-register stopping rules."
  },
  {
    "question": "The control and treatment groups are 55%/45% when 50/50 was intended. What do you suspect?",
    "options": [
      "Noise",
      "Nothing",
      "Bad luck",
      "A sample ratio mismatch: assignment or logging is broken"
    ],
    "correctIndex": 3,
    "explanation": "Fix before analysis."
  },
  {
    "question": "An A/B test shows +0.2% lift, p=0.001 on 10 million users. How do you report it?",
    "options": [
      "Huge win",
      "Statistically significant but tiny; judge practical significance and cost",
      "Insignificant",
      "Ignore"
    ],
    "correctIndex": 1,
    "explanation": "Effect size matters."
  },
  {
    "question": "You ran 20 metric tests and one is p=0.04. What's the concern?",
    "options": [
      "Multiple comparisons: expect about one false positive; correct or predefine primary metrics",
      "Nothing",
      "Increase alpha",
      "It's real"
    ],
    "correctIndex": 0,
    "explanation": "False discovery."
  },
  {
    "question": "Observational data show treated users do better, but they self-selected. What do you use?",
    "options": [
      "Ignore",
      "Causal methods (matching, diff-in-diff) with stated assumptions, ideally a randomized test",
      "Averages",
      "Regression only"
    ],
    "correctIndex": 1,
    "explanation": "Selection bias."
  },
  {
    "question": "A model explains predictions with SHAP. What can you say?",
    "options": [
      "Causes",
      "Accuracy",
      "Which features contributed to a prediction, not that they cause the outcome",
      "Fairness"
    ],
    "correctIndex": 2,
    "explanation": "Interpretability isn't causality."
  },
  {
    "question": "A dashboard loads slowly on 50 million rows. What helps?",
    "options": [
      "Add filters",
      "Pre-aggregate or cache and query only what's needed",
      "Ignore",
      "Remove charts"
    ],
    "correctIndex": 1,
    "explanation": "Performance."
  },
  {
    "question": "Your notebook gives different results when run top to bottom. Likely cause?",
    "options": [
      "Randomness only",
      "Hidden state from out-of-order cells; restart and run all with fixed seeds",
      "Bug in pandas",
      "Hardware"
    ],
    "correctIndex": 1,
    "explanation": "Reproducibility."
  },
  {
    "question": "A hiring model's accuracy is 90% overall but 70% for one group. What next?",
    "options": [
      "Ship",
      "Average",
      "Investigate and mitigate the disparity before deployment",
      "Ignore"
    ],
    "correctIndex": 2,
    "explanation": "Evaluate by subgroup."
  },
  {
    "question": "An executive asks for a one-slide summary. What leads it?",
    "options": [
      "Code",
      "The recommendation and the key evidence",
      "Methods",
      "Raw tables"
    ],
    "correctIndex": 1,
    "explanation": "Answer first."
  }
]
</script>

## Coding challenges

Try these on your own computer, then compare your output. Nothing you write is sent anywhere.

### A/B test z-test

Implement two_proportion_z(conv_a, n_a, conv_b, n_b) using the pooled-proportion z-test and a two-sided p-value from the normal CDF (math.erf). Analyze 200/2000 vs 250/2000 and then a small 200/2000 vs 210/2000 difference.

**Starter code**

```python
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

**Expected output**

```text
A=0.100 B=0.125 lift=25.0%
z=2.50 p=0.0124
significant at 5%: True
small difference p=0.60 significant: False
```

<details><summary>Show solution and explanation</summary>

<pre><code>import math

def two_proportion_z(conv_a, n_a, conv_b, n_b):
    p_a, p_b = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p_a, p_b, z, p_value

p_a, p_b, z, p = two_proportion_z(200, 2000, 250, 2000)
print(&quot;A=%.3f B=%.3f lift=%.1f%%&quot; % (p_a, p_b, (p_b - p_a) / p_a * 100))
print(&quot;z=%.2f p=%.4f&quot; % (z, p))
print(&quot;significant at 5%:&quot;, p &lt; 0.05)
_, _, z2, p2 = two_proportion_z(200, 2000, 210, 2000)
print(&quot;small difference p=%.2f significant: %s&quot; % (p2, p2 &lt; 0.05))
</code></pre>

<p>A 2.5-point lift on 2,000 users per arm is unlikely under no difference, while a 0.5-point lift isn&#x27;t distinguishable from noise. Report effect size and interval too. Verified by running the solution.</p>
</details>

### Least squares and a warning

Implement fit_line(x, y) returning slope, intercept and R-squared by ordinary least squares. Fit a clean linear dataset, predict x=7, and then fit data with one outlier to see how it distorts the line.

**Starter code**

```python
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

**Expected output**

```text
slope=1.974 intercept=0.173 r2=0.9983
prediction for x=7: 13.99
outlier fit slope=4.57 r2=0.70
```

<details><summary>Show solution and explanation</summary>

<pre><code>def fit_line(x, y):
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
print(&quot;slope=%.3f intercept=%.3f r2=%.4f&quot; % (m, c, r2))
print(&quot;prediction for x=7: %.2f&quot; % (c + m * 7))

# Anscombe-style warning: same fit, different shape
x2 = [1, 2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 10, 30]
m2, c2, r22 = fit_line(x2, y2)
print(&quot;outlier fit slope=%.2f r2=%.2f&quot; % (m2, r22))
</code></pre>

<p>The first fit is nearly perfect, while one outlier at x=6 changes the slope sharply and drops R-squared. Always plot your data and inspect residuals. Verified by running the solution.</p>
</details>
