# Question 2 – AI-Based Imputation of Missing Emissions Data 💧

## Problem
Real-world emissions datasets often contain missing values. Traditional imputation methods like mean/median fail to capture underlying patterns. 
This challenge aims to demonstrate a more advanced and realistic imputation approach.

---

## What I Did

### 1. Implemented Advanced Imputation
- I used `sklearn.impute.IterativeImputer` with a `BayesianRidge` estimator.
- It predicts missing values by learning patterns between features, using chained regressions.
- I compared the result with simple mean imputation.

### 2. Data
Simulated 3-column dataset with various NaNs.



## Validation
I printed the original, simple-imputed, and iteratively-imputed datasets.
I verified that no missing values remain after imputation.

## Output Files
iterative_imputer.py: complete implementation with comments.
gan_blueprint.md: GAN architecture design for synthetic emissions data

