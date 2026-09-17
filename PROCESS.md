# Development Process

How `Model_Pipeline.pkl` was built and evaluated, and what to know before
deploying it. Numbers below are reproduced from the actual notebook run
(`credit_card_fraud_detection.ipynb`), not estimated.

## 1. Objective

Classify a credit card transaction as legitimate (`0`) or fraudulent (`1`)
from its elapsed time, amount, and 28 PCA-anonymized features (`V1`–`V28`).

## 2. Dataset

[Kaggle `mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 European cardholder transactions over two days in September 2013.

- **31 columns**: `Time`, `V1`–`V28` (already PCA-transformed upstream — mean
  ≈ 0, so no further scaling needed), `Amount`, `Class`.
- **No missing values** in any column.
- **Class balance**: 99.827% legitimate vs. **0.173% fraud** (492 fraud
  cases total) — the defining challenge of this dataset.

## 3. Preprocessing

A single `ColumnTransformer`, fit inside the pipeline (no leakage):

| Columns | Transform |
|---|---|
| `Time`, `Amount` | `RobustScaler` (resistant to the heavy outliers in transaction amounts) |
| `V1`–`V28` | `passthrough` (already standardized by the dataset's own PCA step) |

## 4. Split

80/20 stratified split (`random_state=42`) to preserve the fraud ratio in
both halves:

| Set | Rows | Fraud cases | Fraud rate |
|---|---|---|---|
| Train | 227,845 | 394 | 0.173% |
| Test | 56,962 | 98 | 0.172% |

## 5. Model selection

Three candidates, each with `class_weight='balanced'` (cost-sensitive
learning — no synthetic oversampling, no SMOTE) evaluated with 3-fold
stratified cross-validation on the training set, scored on **PR-AUC**
(average precision) rather than accuracy, since accuracy on a 99.8%-majority
class is meaningless:

| Model | PR-AUC | Macro-F1 |
|---|---|---|
| Logistic Regression | 0.7515 (± 0.0265) | 0.5508 (± 0.0026) |
| Decision Tree (depth 5) | 0.3370 (± 0.1682) | 0.5537 (± 0.0059) |
| **Random Forest** (depth 5, 50 trees) | **0.7604 (± 0.0172)** | **0.7912 (± 0.0114)** |

Random Forest had both the best PR-AUC and, decisively, the best Macro-F1 —
the single decision tree's high variance (± 0.17 PR-AUC across folds) also
ruled it out as unreliable on its own.

## 6. Hyperparameter tuning

`RandomizedSearchCV` on the Random Forest branch of the pipeline, 5
candidates × 3 folds = 15 fits, scored on PR-AUC:

```
n_estimators:      [50, 100, 200]
max_depth:          [5, 10, 15]
min_samples_split:  [2, 5, 10]
min_samples_leaf:   [1, 2, 4]
```

**Best parameters:** `n_estimators=100, max_depth=10, min_samples_split=5,
min_samples_leaf=1` — **CV PR-AUC 0.8205**.

## 7. Final evaluation (held-out test set, never touched during tuning)

```
                precision    recall  f1-score   support
Legitimate (0)      1.00      1.00      1.00     56864
Fraudulent (1)      0.79      0.83      0.81        98
```

Read plainly: of every 100 transactions the model flags as fraud, about 79
really are; of every 100 actual frauds in the test set, it catches about 83.
That trade-off is a product decision (threshold, cost of a false decline vs.
a missed fraud), not a fixed property of the model — see [Limitations](#9-limitations--honest-caveats).

## 8. Most influential features

Extracted directly from the fitted Random Forest (`feature_importances_`),
not asserted:

| Feature | Importance |
|---|---|
| V15 | 0.216 |
| V13 | 0.139 |
| V5 | 0.114 |
| V11 | 0.084 |
| V12 | 0.079 |

Because `V1`–`V28` are PCA components, these are anonymized directions in
feature space, not interpretable merchant/spending categories.

## 9. Limitations & honest caveats

- **Single benchmark dataset.** This is the standard academic
  fraud-detection benchmark, not a live card network feed. Treat this repo
  as a demonstration of the modeling and deployment pipeline, not a
  production fraud system.
- **No drift monitoring.** Fraud patterns shift over time; this pipeline
  doesn't retrain or alert on drift.
- **Default 0.5 decision threshold.** `/predict` returns both the class and
  the raw `fraud_probability` so a consumer can apply a different threshold
  for their own precision/recall trade-off instead of trusting the default.
- **Free-tier cold starts.** Render's free plan sleeps a service after 15
  minutes idle; the first request afterward can take 30–60 seconds while it
  wakes back up.

## 10. Deployment gotcha worth recording

`Model_Pipeline.pkl` was produced with **scikit-learn 1.6.1**. Newer
scikit-learn (1.7+) cannot unpickle it — it raises
`AttributeError: Can't get attribute '_RemainderColsList'` on
`ColumnTransformer`, a private helper class that was removed/relocated in
later releases. `requirements.txt` pins `scikit-learn==1.6.1` deliberately;
bumping it without retraining and re-serializing the pipeline will break
`/predict` on the next deploy.

## 11. Artifact

The saved object is the **entire pipeline** (`preprocessor` + tuned
`classifier`), via `joblib.dump`, so inference code passes a raw 30-column
row straight in — no separate manual scaling step to keep in sync.

## 12. Demo UI data

`V1`–`V28` aren't values a person could reasonably type from scratch, so
the web UI's "Load a real transaction" and "Randomize" controls needed
actual numbers to draw on. Rather than inventing a plausible-looking
"fraud example" — which would misrepresent this dataset's central fact,
that fraud is 0.17% of it — the sample transactions are five real rows
from the dataset (`df.head()`), and the randomizer draws from the real
per-column mean/standard deviation in `df.describe()`. Both are labeled in
the UI for what they are, and neither is presented as a real fraud case.
