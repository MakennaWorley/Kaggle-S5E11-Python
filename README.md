# Kaggle Playground S5E11: Predicting Loan Payback
CMPT 311 Machine Learning (Fall 2025) - Final Project

**Kaggle Episode:** Season 5, Episode 11

**Competition Link:** [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

---

## 1. Project Overview
This repository contains the final capstone project for **CMPT 311 Machine Learning**. The objective was to compete in a Kaggle playground episode and build a complete machine learning pipeline to predict the that episode's goal. In my case the probability that a borrower will pay back a loan.

**The Challenge:**
Using a synthetic dataset generated from real-world loan data, the goal is to predict the target variable `loan_paid_back`

## 2. Repository Structure & File Guide

The project is divided into two phases of development, consistent with the course and Kaggle rules:

```
├── data/                       # Dataset files (can be downloaded from Kaggle)
├── submissions/                # Submission files (not posted)
├── s5e11.ipynb                 # PHASE 1: Initial Experimentation (Sandbox)
│                               # - Exploratory and baseline implementation
│
├── s5e11 v2.ipynb              # PHASE 2: Final Project (Presentation Code)
│                               # - Advanced Feature Engineering
│                               # - Extensive Model Comparison (Softmax, LDA, RF, GB, etc. all via sklearn)
│                               # - Hyperparameter Tuning & PCA experiments
│                               # - Final Visualizations & Confusion Matrices
│
├── presentation.pdf            # Final presentation slides
└── README.md                   # Project documentation
```

## 3. Data Understanding & EDA

The dataset consists of approximately 593,994 rows with a mix of numerical and categorical features.

- Target Variable: `loan_paid_back`
- Key Features: `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`.
- Distributions: Analyzed via histograms to understand skewness (e.g., Income was highly right-skewed).
- Correlations: Pairwise relationships were examined to identify multicollinearity.

## 4. Methodology

### Preprocessing & Feature Engineering

Beyond standard cleaning, I engineered four key features to better capture financial risk:

- Loan-to-Income Ratio: Quantifies the financial burden relative to capacity ($\text{loan\_amount} / \text{annual\_income}$).
- Disposable Income Estimate: Proxies monthly cash flow by adjusting income for existing debt ($\text{annual\_income} \times (1 - \text{DTI})$).
- Risk Interaction: Multiplies DTI by Interest Rate to highlight compounded vulnerability.
- Log Transformation: Applied $\ln(x+1)$ to annual_income to reduce the impact of outliers and skewness.

Standard techniques included:

- Encoding: One-Hot Encoding for categorical variables.
- Scaling: Standardization using StandardScaler for numerical inputs.

### Model Comparison (sklearn)

I implemented and compared a wide range of classifiers, testing both "Categorical" (subset) and "Full" feature sets:

- Linear Models: Logistic Regression, Softmax Regression, and Linear Discriminant Analysis (LDA).
- Tree-Based Models: Decision Trees (Gini & Entropy), Random Forest, and Gradient Boosting.
- Support Vector Machines (SVM): Tested Linear, Polynomial, and RBF kernels.
- Generative Models: Naive Bayes (Gaussian & Bernoulli) and QDA.
- Dimensionality Reduction: Tested Principal Component Analysis (PCA) combined with various classifiers.

Optimization Strategy: Top-performing models (Gradient Boosting, Random Forest) underwent Hyperparameter Tuning using RandomizedSearchCV to optimize estimators, learning rates, and tree depth.

## 5. Results & Evaluation

Models were evaluated based on Accuracy, ROC-AUC, and Average Precision. I also performed an overfitting analysis by calculating the gap between Train and Test AUC.

![results.png](results.png)

### Key Findings:

- Best Performer: The Tuned Gradient Boosting model achieved the highest results with an AUC of ~0.922 and Accuracy of ~0.876. It successfully captured non-linear relationships that simpler models missed.
- Linear Separability: Linear models like LDA performed surprisingly well (Accuracy: 0.882), suggesting the dataset has strong linear components, though they lagged slightly in probability ranking (AUC) compared to boosting.
- PCA Impact: Models utilizing PCA consistently scored lower than their full-feature counterparts, indicating that dimensionality reduction resulted in a loss of critical predictive signal.
- Feature Importance: The engineered feature grade_subgrade offered negligible predictive value, as models excluding it performed nearly identically to those including it.

### Confusion Matrix Analysis (Best Model):
| | Predicted Did Not Pay | Predicted Paid Back | TOTAL True Count |
| :--- | :---: | :---: | :---: |
| **True Did Not Pay** | 28,198 (TP) | 7,899 (FN) | 36,097 |
| **True Paid Back** | 14,222 (FP) | 127,880 (TN) | 142,102 |
| **TOTAL Predicted Count** | 42,420 | 135,779 | 178,199 (Total Samples) |

Visualizations, including ROC Curves and Precision-Recall comparisons for all models, can be found in the final notebook.