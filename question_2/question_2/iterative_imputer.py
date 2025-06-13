# ===============================
# Advanced Imputation with IterativeImputer
# Author: Rayen Bakini
# ===============================

# Importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Required to enable IterativeImputer (since it's still experimental in scikit-learn)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

# --- 1. Creating a small dataset with missing values ---
# I'm simulating a small dataset with missing values (NaN) to test imputation methods
X = np.array([
    [10., 2., 30.],
    [4., np.nan, 60.],
    [np.nan, 8., 90.],
    [10., 5., np.nan]
])

# Converting the NumPy array into a Pandas DataFrame for easier handling
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
print(" Original Data with Missing Values:")
print(df)

# --- 2. Simple method: mean imputation ---
# Here I fill missing values using the column-wise mean (basic approach)
simple_imputer = SimpleImputer(strategy='mean')
X_simple = simple_imputer.fit_transform(X)
df_simple = pd.DataFrame(X_simple, columns=df.columns)

print("\n Simple Mean Imputation:")
print(df_simple)

# --- 3. Advanced method: IterativeImputer with BayesianRidge ---
# Now I'm using IterativeImputer with a BayesianRidge model to estimate missing values.
# This method is smarter as it learns relationships between features
iter_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
X_iterative = iter_imputer.fit_transform(X)
df_iterative = pd.DataFrame(X_iterative, columns=df.columns)

print("\n Iterative Imputation with BayesianRidge:")
print(df_iterative)
