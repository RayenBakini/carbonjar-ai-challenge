# Question 12 – Addressing Class Imbalance ⚖️

## Problem
Anomalies (class 1) are very rare in our data (5%), so a standard classifier tends to ignore them and report high accuracy but very low recall on the minority class.

---

## What I Did

1. **Baseline model (no SMOTE):**
   - Simulated a dataset with 95% negative (0) / 5% positive (1).
   - Trained a `LogisticRegression` on the original imbalanced training set.
   - Measured metrics on the test set: Accuracy, Precision, Recall, F1.

2. **Apply SMOTE:**
   - Used `imblearn.over_sampling.SMOTE` to oversample the minority class in the training data.
   - Retrained the same `LogisticRegression` on the balanced set.
   - Measured the same metrics again on the unchanged test set.

---

## Results

Before SMOTE:
  Accuracy : 0.953
  Precision: 0.857
  Recall   : 0.316
  F1 Score : 0.462


After SMOTE:
  Accuracy : 0.887
  Precision: 0.286
  Recall   : 0.526
  F1 Score : 0.370

