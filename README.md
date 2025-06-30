# Employee Attrition Prediction & Comparative Analysis of Resampling Techniques

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)

## 🔍 Problem Overview

Predicting the employees who are prone to attrition allows the company to take proactive measures to  
improve retention.  
Additionally, datasets used for machine learning models are often unbalanced and it becomes crucial to  
adopt correct balancing techniques for optimal results.

---

## 🎯 Objectives

- Predict employee attrition using 6 basic supervised learning models- Logistic Regression, Random Forest Classifier, KNN, XGB Classifier, SVC, Decision Tree Classifier.
- Perform comparative analysis of **8 resampling techniques** to handle class imbalance.
- Identify the best-performing model and resampling combination using evaluation metrics.

---
## 📊 Dataset

- **Source**: [IBM HR Analytics Employee Attrition Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size**: 1470 records × 35 features
  - 9 categorical features
  - 26 numerical features
  - Target variable: `Attrition`

---
## 🔬 Novelty

This project stands out by conducting a **comparative analysis of eight different resampling techniques**  to address class imbalance — a common challenge in HR analytics. Unlike many existing works that rely on a single balancing method or overlook imbalance entirely, this project systematically evaluates under-sampling, over-sampling, and hybrid approaches to determine the most effective strategy.

The combination of **Random Forest** and **Random Oversampling** yielded exceptional results (as obtained after performing cross validation of the combination):
- **Accuracy**: 98.18%
- **AUC-ROC**: 99.99%

> 🔥 These scores are significantly higher than most publicly available implementations on the IBM HR dataset.
---

## ⚙️ Methodology

### 1. **Data Preprocessing & Feature Engineering**
- Applied Chi-Square test for categorical feature relevance.
- Performed correlation analysis for feature selection.
- Encoded categorical variables using **Label Encoding**.
- Applied **Standard Scaling** and **Principal Component Analysis (PCA)** for dimensionality reduction.

### 2. **Resampling Techniques Used**
- **Under-sampling**:
  - Random Under Sampling
  - Tomek Links
  - Edited Nearest Neighbors (ENN)
- **Over-sampling**:
  - Random Over Sampling
  - SMOTE
  - ADASYN
- **Hybrid Techniques**:
  - SMOTE + Tomek Links
  - SMOTE + ENN

### 3. **Models Used**
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors
- XGBoost Classifier
- Support Vector Classifier (SVC)
- Decision Tree Classifier

### 4. **Model Selection & Validation**
- **Train/Test Split**: 80/20
- **Cross-validation** applied for robustness
- **RandomizedSearchCV** used for hyperparameter tuning

### 5. **Evaluation Metrics**
- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC Score
- ROC Curves

---

## 🏆 Results

- **Best Model**: Random Forest Classifier
- **Best Resampling**: Random Over Sampling
- **Performance** (as obtained after performing cross validation for the combination):
  - Accuracy: **98.18%**
  - AUC-ROC: **99.99%**

---


