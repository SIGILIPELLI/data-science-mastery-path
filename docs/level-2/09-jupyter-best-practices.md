# 09 · Jupyter Notebook Best Practices

Notebooks are great for exploration and terrible for reproducibility if
you don't discipline your workflow. This module covers the habits that
keep a notebook trustworthy and shareable, not just "worked once on my
machine."

## Restart and run all, always

The single most valuable habit: before trusting a notebook's output (and
definitely before sharing it), use **Kernel → Restart & Run All**. Cells
run out of order all the time during exploration — you delete a cell that
defined a variable another cell still depends on, or you rerun cell 12
after editing cell 3. The kernel's *in-memory* state can silently diverge
from what a fresh top-to-bottom execution would produce.

```python
# Anti-pattern: this works right now because you ran an earlier cell
# out of order, but a fresh "Restart & Run All" would raise NameError.
adjusted = raw_value * scaling_factor   # scaling_factor defined three cells below
```

If "Restart & Run All" fails or gives different numbers than what's
currently displayed, the notebook is lying about its own results.

## One cell, one idea

```python
# Bad: five unrelated things crammed into one cell
import pandas as pd
df = pd.read_csv("sales.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["amount"])
summary = df.groupby("region")["amount"].sum()
print(summary)
```

```python
# Better: each cell is independently re-runnable and debuggable
df = pd.read_csv("sales.csv")
```

```python
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["amount"])
```

```python
summary = df.groupby("region")["amount"].sum()
summary
```

Small cells make it obvious *where* something broke, let you inspect
intermediate state without re-running everything, and produce a cleaner
diff when the notebook is version-controlled.

## Keep heavy logic in `.py` files, not cells

```python
# analysis/cleaning.py
import pandas as pd

def load_clean_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["amount"])
```

```python
# In the notebook:
from analysis.cleaning import load_clean_sales

df = load_clean_sales("sales.csv")
df.head()
```

Functions in a `.py` module are testable with `pytest`, importable from
multiple notebooks without copy-paste, and reviewable in a normal code
diff — notebook JSON diffs are close to unreadable. A good rule: if a cell
has more than ~15 lines of logic (not display code), it probably belongs
in a module.

## Pin your environment

```text
# requirements.txt or environment.yml — commit this alongside the notebook
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
```

"It works on my machine" almost always means an unpinned dependency
changed behavior. Recreate environments with
`pip install -r requirements.txt` (or `conda env create -f environment.yml`)
so a colleague — or you, in six months — gets the same results.

## Clear outputs before committing (usually)

```bash
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

Committing a notebook with large embedded outputs (plots as base64 images,
long DataFrame dumps) bloats the git history and makes diffs unreadable.
The common exception: a *final*, presentation-ready notebook meant to be
viewed directly on GitHub — there, keep outputs so reviewers don't need to
re-run anything.

## Seed randomness and record versions

```python
import random, numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(pd.__version__, np.__version__)
```

Any notebook using `np.random`, `train_test_split`, or a stochastic model
should fix a seed — otherwise "I got a different number this time" becomes
a recurring, unresolvable question. Printing library versions at the top
of the notebook (or better, in the pinned requirements file) closes the
loop on "what exactly produced this output."

## A minimal notebook checklist

- [ ] Restart & Run All succeeds top to bottom with no errors
- [ ] Random seeds are set wherever randomness is used
- [ ] Heavy logic lives in importable functions, not sprawling cells
- [ ] Dependencies are pinned in a committed requirements file
- [ ] Outputs are cleared before committing (unless it's a final report)
- [ ] The first markdown cell states the notebook's purpose and data source

## Cheat sheet

| Practice | Why |
|---|---|
| Restart & Run All before trusting results | Catches out-of-order execution bugs |
| One idea per cell | Isolates failures, cleaner diffs |
| Move logic to `.py` modules | Testable, reusable, reviewable |
| Pin dependencies | Reproducible environment |
| Clear outputs before commit | Small, readable git history |
| Seed randomness | Reproducible numbers |

## Exercise

Take any notebook you've written with more than 10 cells. Run
Kernel → Restart & Run All. If it fails or produces different numbers,
identify the out-of-order dependency that caused it, and fix the cell
ordering so a fresh run is deterministic and correct.
