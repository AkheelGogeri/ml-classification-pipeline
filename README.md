# Obesity Level Classification — End-to-End ML Pipeline

MSc Applied AI | COMP634 Assessment 1 | University of Liverpool (2024/25)
Solo Project

---

## Overview

A complete supervised machine learning pipeline to classify obesity levels across 7 categories based on lifestyle, dietary, and demographic features. Four classifiers were implemented, tuned, and benchmarked under identical conditions using a leakage-free cross-validation protocol. The best model achieved 95.5% test accuracy.

---

## Problem Statement

Predict an individual's obesity category (Insufficient Weight, Normal Weight, Overweight I–II, Obesity I–III) from numerical and categorical features including age, weight, height, meal frequency, physical activity, and technology usage. Modelled as a 7-class supervised classification task.

---

## Dataset

- **Target variable:** ObesityLvl — 7 classes
- **Features:** Numerical (Age, Weight, Height, Meals, VegMeals, WaterQ, PhysAct, TechUsage) + Categorical (Gender, Transportation, Family History, etc.)
- **Split:** 80% training / 20% test, stratified sampling to preserve class balance

---

## Pipeline

### Data Cleaning & Preprocessing
- Removed duplicate rows and unrealistic biological outliers (Age > 100, Height > 2.2m, Weight > 200kg) identified via boxplot analysis
- Imputed missing values using median (numerical) and most-frequent (categorical) strategies
- One-hot encoded categorical features
- Standardised numerical features within pipeline to prevent data leakage
- Retained minor skewness to preserve data realism

### Model Selection
Four classifiers were implemented and tuned via GridSearchCV with 5-fold Stratified Cross-Validation (F1-macro as optimisation metric):

| Model | Mean CV F1-Macro | Std Dev | Best Parameters |
|---|---|---|---|
| SVM (Linear) | 0.961 ± 0.012 | Very stable | kernel='linear', C=10, gamma='scale' |
| Random Forest | 0.942 ± 0.010 | Slightly lower | n_estimators=300, max_depth=20 |
| Decision Tree | 0.934 ± 0.018 | Slightly overfit | max_depth=10, criterion='gini' |
| KNN | 0.860 ± 0.025 | Sensitive to scaling | n_neighbors=5, weights='distance' |

---

## Results — Best Model: SVM (Linear Kernel)

| Metric | Cross-Validation | Test Set | Difference |
|---|---|---|---|
| Accuracy | 0.961 | 0.955 | 0.006 |
| F1-macro | 0.960 | 0.953 | 0.007 |
| F1-weighted | 0.961 | 0.955 | 0.006 |

The <1% gap between CV and test metrics confirms excellent generalisation — the model is neither overfitting nor underfitting.

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Insufficient Weight | 0.96 | 1.00 | 0.98 |
| Normal Weight | 0.98 | 0.93 | 0.95 |
| Overweight I | 0.91 | 0.91 | 0.91 |
| Overweight II | 0.90 | 0.93 | 0.92 |
| Obesity Type I | 0.99 | 0.96 | 0.97 |
| Obesity Type II | 0.97 | 1.00 | 0.98 |
| Obesity Type III | 1.00 | 0.98 | 0.99 |

Minor confusion observed between Overweight I and Overweight II (adjacent categories). All other classes show strong separation.

### Top Feature Influences (Linear SVM Coefficients)
1. Weight — Very High
2. Height — High
3. Meals — Moderate
4. VegMeals — Moderate
5. PhysAct — Moderate

Body weight and height dominate predictions, aligning with medical expectations. Lifestyle features contribute meaningfully at the margin.

---

## Tech Stack

- Python, scikit-learn, pandas, NumPy, matplotlib, seaborn
- GridSearchCV, StratifiedKFold, Pipeline API

---

## Repository Structure

```
ml-classification-pipeline/
├── notebook.ipynb       # Full pipeline: EDA, preprocessing, modelling, evaluation
├── report.pdf           # Written analysis and discussion
└── README.md
```

---

## Key Findings

- SVM with a linear kernel outperformed all other models — strong when classes are linearly separable after standardisation
- Random Forest was a close second, confirming non-linear patterns exist in the data
- The preprocessing pipeline (built inside sklearn Pipeline) ensured no data leakage across CV folds
- KNN was most sensitive to the curse of dimensionality despite distance-weighted voting

---

## Academic Context

- **Module:** COMP634 — Applied AI (Assessment 1)
- **Institution:** University of Liverpool, Department of Computer Science
- **Year:** 2024/25

---

## Author

**Akheel R Gogeri**
MSc Artificial Intelligence with Data Science — University of Liverpool
[LinkedIn](https://linkedin.com/in/akheel-gogeri) | [GitHub](https://github.com/AkheelGogeri)

