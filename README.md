# Credit Card Fraud Detection with Explainable AI

A machine learning application that detects fraudulent credit card transactions using **XGBoost** and provides transparent predictions through **SHAP Explainable AI**. This project compares multiple machine learning models, engineers transaction-based features, and deploys the final model as an interactive **Streamlit** web application.

**Team:** AI4All Ignite – Group 12D

---

## Live Demo

**Streamlit App:** *( https://ai4all-fraud-detectiongit-astg3tefygpdzk9nclhtbq.streamlit.app/ )*

**Demo Video:** (--demo video link–)

---

# Project Overview

Credit card fraud results in billions of dollars in financial losses each year, making accurate fraud detection a critical challenge for financial institutions. Because fraudulent transactions represent only a small fraction of all transactions, models must identify fraud while minimizing false positives.

This project investigates whether explainable AI techniques, specifically **SHAP (SHapley Additive exPlanations)** and **feature importance analysis**, can improve fraud detection by identifying the most informative engineered features.

We trained and evaluated multiple machine learning models, ultimately selecting **XGBoost** as the best-performing model after applying:

- Feature engineering
- SMOTE
- Hyperparameter tuning
- Threshold optimization

To improve transparency, SHAP explanations are integrated into a Streamlit application so users can understand why a transaction was classified as fraudulent or legitimate.

---

# Project Highlights

- End-to-end fraud detection pipeline
- Exploratory Data Analysis (EDA)
- Feature engineering
- Logistic Regression, Random Forest, and XGBoost comparison
- SMOTE for class imbalance
- Threshold tuning
- SHAP Explainable AI
- Streamlit deployment

---

#  Dataset

**Sparkov Credit Card Fraud Detection Dataset**

https://www.kaggle.com/datasets/kartik2112/fraud-detection

- Simulated U.S. credit card transactions
- January 2019 – December 2020

---

#  Data Preprocessing

The preprocessing pipeline included:

- Removing unnecessary columns
- Handling missing values
- Removing duplicate records
- Outlier analysis
- Encoding categorical variables
- Train/test split
- SMOTE oversampling
- Feature scaling
- Threshold optimization

---

# Feature Engineering

The final model was trained using the **15 most important engineered features**, selected using Feature Importance and SHAP analysis.

| Feature | Description |
|---------|-------------|
| amt_vs_avg | Transaction amount relative to historical average |
| card_avg_amt | Historical average transaction amount |
| trans_count_24h | Number of transactions in previous 24 hours |
| secs_since_last | Time since previous transaction |
| hour | Transaction hour |
| hour_sin | Cyclical hour feature |
| hour_cos | Cyclical hour feature |
| is_night | Nighttime indicator |
| age | Cardholder age |
| amt_bin_0 | Lowest transaction amount range |
| amt_bin_1 | Second amount range |
| amt_bin_2 | Third amount range |
| amt_bin_3 | Fourth amount range |
| amt_bin_4 | Highest amount range |
| category | Encoded merchant category |

---

# 🤖 Model Development

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

## Techniques Used

- Feature engineering
- Categorical encoding
- SMOTE
- GridSearchCV
- 10-fold Cross Validation
- SHAP analysis
- Feature importance
- Threshold optimization

---

# Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| Logistic Regression | 99.00% | 32.00% | 60.00% | 42.00% | 0.94 |
| Random Forest | 100.00% | 66.00% | 62.00% | 64.00% | 0.98 |
| XGBoost (Baseline) | 99.00% | 47.00% | 71.00% | 57.00% | 0.98 |
| **XGBoost (Final)** | **99.53%** | **57.89%** | **66.76%** | **62.00%** | **0.9792** |

## Key Findings

- Logistic Regression improved after feature engineering.
- Random Forest achieved the highest precision.
- XGBoost achieved the highest recall.
- SHAP identified the most influential engineered features.
- The final optimized XGBoost achieved:
  - **99.53% Accuracy**
  - **57.89% Precision**
  - **66.76% Recall**
  - **62.00% F1-score**
  - **0.9792 ROC-AUC**

---

# Explainable AI (SHAP)

Feature Importance and SHAP were used to improve model transparency.

## Feature Importance

Feature importance ranking identified the **15 most informative engineered features** for the final XGBoost model.

*(Feature Importance plot )*

## SHAP Explainability

For every prediction, SHAP shows:

- Which features influenced the prediction
- Whether each feature increased or decreased fraud likelihood
- The magnitude of each feature's contribution

*(SHAP explanation image)*

---

# Streamlit Application

The deployed application allows users to upload a CSV containing a single transaction.

### Workflow

1. Upload transaction CSV
2. Automatic feature engineering
3. Fraud prediction
4. SHAP explanation generation
5. Display prediction and explanation

*(Insert application screenshots here)*

---

# Repository Structure

```text
fraud_detection_streamlit/

├── app.py
├── models/
│   ├── best15_xgb2.json
│   ├── model_info.pkl
│   ├── card_database.pkl
│   └── category_dict.pkl
│
├── notebooks/
│   └── CreditCardFraudDetection.ipynb
│
├── utils/
│   ├── constants.py
│   ├── descriptions.py
│   ├── feature_engineering.py
│   ├── load_resources.py
│   └── shap_utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

#  Installation

```bash
git clone https://github.com/yourusername/fraud_detection_streamlit.git

cd fraud_detection_streamlit

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

#  Usage

```bash
streamlit run app.py
```

---

# Team Members & Contributions

| Team Member | Contributions |
|-------------|--------------|
| **Ashriya Tuladhar** | Preprocessed merchant and geographic features (merchant, job, lat, long, merch_lat, merch_long); engineered transaction-based features; developed, tuned, and evaluated the XGBoost model; performed feature importance and SHAP analysis; designed, implemented, and deployed the Streamlit application. |
| **Mohamed Awad** | Preprocessed population and location features (city_pop, zip); contributed to feature engineering; developed and evaluated the XGBoost and Random Forest models; performed feature importance and SHAP analysis. |
| **Hadi Malik** | Preprocessed categorical and demographic features (category, gender, dob); developed and evaluated the Random Forest model; performed feature importance and SHAP analysis. |
| **Brownkaine Forchick** | Preprocessed transaction and time-related features (trans_date_trans_time, cc_num, unix_time); developed and evaluated the Logistic Regression model; performed feature importance and SHAP analysis. |
| **Aliyu Aliyu** | Preprocessed location features (city, state); developed and evaluated the Logistic Regression model; performed feature importance and SHAP analysis. |

---

# Future Improvements

- Improve precision and recall
- Explore deep learning models
- Evaluate on real-world financial datasets
- Build a REST API
- Model drift monitoring
- Real-time transaction streaming

---

# Acknowledgements

This project was completed as part of the **AI4All Ignite Program**.

We thank our mentors, instructors, and teammates for their guidance and collaboration throughout this project.
