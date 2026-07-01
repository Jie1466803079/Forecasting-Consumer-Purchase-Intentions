# Forecasting Consumer Purchase Intentions

> A binary-classification study that predicts whether a customer will purchase, using their gender, age, and estimated salary, and compares a Naive Bayes classifier against Logistic Regression.

`scikit-learn` · `imbalanced-learn (SMOTE)` · `Naive Bayes & Logistic Regression`

## Problem
The goal is to run an ETL process to clean the data, extract insights through visualization, and build models that predict whether a user will exhibit purchasing behaviour to support the company's strategic decisions. The stated stakeholders are the company and its clients. The initial hypotheses are that a user's gender, age, and salary may affect purchasing behaviour.

## Approach
The project follows a Big Data Analytics Lifecycle framework:

1. **Data** — Load `test2_data.csv`: 400 records and 5 columns (`User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased`).
2. **ETL / preprocessing** — Confirm no duplicates, no missing values, and no outliers in `Age`/`EstimatedSalary` (via boxplots); one-hot encode `Gender`; explore distributions (histograms) and a correlation heatmap.
3. **Feature selection** — Keep `Gender_Male`, `Age`, and `EstimatedSalary`, dropping `Gender_Female` to avoid multicollinearity.
4. **Class imbalance** — The target is imbalanced (257 non-purchasers vs. 143 purchasers); addressed with SMOTE oversampling (Naive Bayes) and `class_weight='balanced'` (Logistic Regression).
5. **Split & scale** — Shuffle the data, split 60/20/20 into training / validation / test sets, and standardize features with `StandardScaler`.
6. **Models** — Train `GaussianNB` (Naive Bayes) and `LogisticRegression`.
7. **Evaluation** — Compare on accuracy, precision, recall, F1, and ROC-AUC, plus confusion matrices, on both validation and test sets.

## Results
Metrics are reported exactly as printed in the notebook (2 decimal places).

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| GaussianNB | Validation | 0.85 | 0.72 | 0.88 | 0.79 | 0.86 |
| GaussianNB | Test | 0.95 | 0.91 | 0.97 | 0.94 | 0.95 |
| Logistic Regression | Validation | 0.84 | 0.69 | 0.92 | 0.79 | 0.86 |
| Logistic Regression | Test | 0.86 | 0.81 | 0.84 | 0.83 | 0.86 |

On the test set the Naive Bayes (GaussianNB) classifier scored highest (accuracy 0.95 vs. 0.86 for Logistic Regression). Note the gap between its validation (0.85) and test (0.95) accuracy on this small dataset.

Fitted Logistic Regression equation:
```
logit(P) = -0.512127983256144 + 0.17779686582053025 * Gender_Male + 2.103933200467163 * Age + 1.0358402974253895 * EstimatedSalary
```

Stated future work: explore Random Forest, Decision Tree, and Gradient Boosting Machines (GBM).

## Repository structure
```
Forecasting-Consumer-Purchase-Intentions/
├── Forecasting Consumer Purchase Intentions.ipynb   # full analysis: ETL, EDA, modelling, evaluation
└── README.md
```
Note: the notebook reads `test2_data.csv`, which is not included in the repository.

## Tech
- **Language:** Python (Jupyter Notebook)
- **Data & compute:** pandas, NumPy
- **Modelling:** scikit-learn (`GaussianNB`, `LogisticRegression`, `StandardScaler`, `train_test_split`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`), imbalanced-learn (`SMOTE`)
- **Visualization:** matplotlib, seaborn
