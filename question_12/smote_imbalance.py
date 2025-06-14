# ===============================
# Addressing Class Imbalance with SMOTE
# Author: Rayen Bakini
# ===============================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# 1. Simulate an imbalanced dataset (95% class 0, 5% class 1)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.95, 0.05],
    flip_y=0,
    random_state=42
)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train a baseline Logistic Regression without SMOTE
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(" Before SMOTE:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.3f}")

# 4. Apply SMOTE to oversample the minority class in training set
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 5. Retrain the model on the balanced data
model_sm = LogisticRegression(solver='liblinear')
model_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = model_sm.predict(X_test)

print("\n After SMOTE:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_sm):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred_sm):.3f}")
print(f"  Recall   : {recall_score(y_test, y_pred_sm):.3f}")
print(f"  F1 Score : {f1_score(y_test, y_pred_sm):.3f}")
