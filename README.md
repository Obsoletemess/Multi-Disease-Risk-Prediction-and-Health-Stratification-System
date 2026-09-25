# Multi-Disease Risk Prediction and Health Stratification System

An explainable clinical machine learning pipeline designed to simultaneously predict cardiovascular risk, metabolic risk, and an overall health severity tier using routine checkup data[cite: 1].

## 📊 Dataset
* **Source:** CDC National Health and Nutrition Examination Survey (NHANES)[cite: 1].
* **Size:** 15,560 patient records spanning 351 variables[cite: 1].
* **Features:** Integrates laboratory biochemistry panels, physical examination measures, and self-reported questionnaire data (e.g., lifestyle, diagnostic history)[cite: 1].

## 🏗️ Architecture & Pipeline
This project is structured across five core stages to ensure high accuracy and clinical interpretability[cite: 1]:

1. **Preprocessing & Redundancy Purge:** Derives multi-tier target labels, balances classes using SMOTE, and removes multicollinearity via a two-stage statistical filter (dropping features with Spearman |ρ| > 0.85, then iteratively dropping features until VIF ≤ 5.0)[cite: 1].
2. **Parallel Feature Scoring:** Scores surviving features under three independent methods: ANOVA F-test, Mutual Information, and LASSO (L1) regularization[cite: 1].
3. **Model Tournament & Deep Learning:** Trains and evaluates diverse base learners, including Random Forest, XGBoost, LightGBM, CatBoost, and a PyTorch-backed deep tabular FT-Transformer[cite: 1].
4. **Leak-Free Stacking:** Combines 5-fold out-of-fold (OOF) predictions from the base models using a regularized Logistic Regression meta-learner to maximize AUROC[cite: 1].
5. **Explainability & Deployment:** Wraps the ensemble model in a Streamlit dashboard that utilizes SHAP to generate real-time risk scores paired with per-patient feature attribution and global driver rankings[cite: 1].

## 🛠️ Tech Stack
* **Language:** Python (v3.12+)[cite: 1]
* **Data Processing & Statistics:** Pandas (v2.2+), NumPy (v2.0+), SciPy (v1.13+), Statsmodels (v0.14+)[cite: 1]
* **Machine Learning:** Scikit-Learn (v1.5+), XGBoost (v2.0+), LightGBM (v4.3+), CatBoost (v1.2+)[cite: 1]
* **Deep Learning:** PyTorch (v2.4+)[cite: 1]
* **Explainable AI (XAI) & UI:** SHAP (v0.45+), Streamlit (v1.35+)[cite: 1]
