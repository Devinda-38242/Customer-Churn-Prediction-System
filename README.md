# CHURN INTELLIGENCE

## AI-Powered Customer Churn Prediction & Business Intelligence for Telecommunications

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **CHURN INTELLIGENCE** is an end-to-end Machine Learning and Explainable AI platform designed to help telecommunications businesses identify customers at risk of churn, understand the factors driving that risk, and support data-driven retention decisions.

---

## Overview

Customer churn is one of the most important challenges faced by telecommunications companies.

Acquiring new customers can be significantly more expensive than retaining existing customers. A reliable churn prediction system can therefore help businesses identify high-risk customers early and take appropriate retention actions.

**CHURN INTELLIGENCE** goes beyond a simple churn classification model.

The platform combines:

* Machine Learning-based churn prediction
* Churn probability estimation
* Customer risk classification
* SHAP-based Explainable AI
* Customer-specific risk analysis
* Retention recommendations
* Batch customer prediction
* Batch analytics
* Business intelligence dashboards
* Interactive Streamlit interface

The goal is to answer three important business questions:

> **Who is likely to churn?**

> **Why are they likely to churn?**

> **What can the business do about it?**

---

# Project Scope

### Current Industry: Telecommunications

The current version of CHURN INTELLIGENCE is specifically designed for **telecommunications customer churn prediction**.

The underlying Machine Learning model has been trained using telecommunications customer attributes such as:

* Contract type
* Customer tenure
* Monthly charges
* Total charges
* Internet service
* Phone service
* Online security
* Online backup
* Device protection
* Tech support
* Streaming services
* Payment method
* Paperless billing
* Customer demographics

### Important Limitation

The current model should **not** be directly applied to customer datasets from other industries such as banking, insurance, healthcare, retail, or SaaS.

Different industries have different customer behaviors, features, churn definitions, and business processes.

Supporting additional industries would require separate datasets, feature engineering pipelines, models, evaluation, explainability logic, and business rules.

---

# Key Features

## 1. Customer Churn Prediction

The system evaluates customer information and predicts whether the customer is likely to churn.

The prediction is accompanied by a churn probability, allowing the system to represent customer risk more meaningfully than a simple Yes/No prediction.

---

## 2. Customer Risk Analysis

Customers are categorized according to their estimated churn risk.

The system provides information such as:

* Churn probability
* Risk level
* Customer profile
* Important customer characteristics
* Contributing risk factors

This allows users to quickly identify customers requiring attention.

---

## 3. Explainable AI with SHAP

CHURN INTELLIGENCE uses **SHAP (SHapley Additive exPlanations)** to make Machine Learning predictions more understandable.

Instead of providing only:

```text
HIGH CHURN RISK
```

the system can identify the factors contributing to the prediction.

Examples of potentially influential factors include:

* Month-to-month contracts
* Short tenure
* Higher monthly charges
* Fiber optic internet service
* Lack of technical support
* Lack of online security services
* Payment method
* Other customer characteristics

SHAP is used for both:

### Global Explainability

Understanding which features have the greatest influence across the model.

### Individual Explainability

Understanding why a specific customer received a particular churn prediction.

---

# 4. Retention Recommendations

The system connects predicted churn risk with customer characteristics and model insights to generate actionable retention recommendations.

Examples include:

* Contract upgrade opportunities
* Personalized offers
* Service bundle recommendations
* Technical support assistance
* Security service recommendations
* Loyalty-focused retention strategies

The objective is to transform:

```text
Prediction
    ↓
Risk Factor
    ↓
Recommendation
    ↓
Potential Retention Action
```

---

# 5. Batch Prediction

CHURN INTELLIGENCE supports analysis of multiple customers through CSV upload.

Users can upload a telecommunications customer dataset and generate predictions for multiple customers.

Batch processing can provide:

* Customer-level churn predictions
* Churn probabilities
* Risk classifications
* Risk distributions
* Aggregate churn statistics
* Business-level insights

### Dataset Requirement

The uploaded dataset must contain the customer attributes required by the trained Telco preprocessing and prediction pipeline.

The system is **not a generic CSV churn predictor**.

---

# 6. Batch Analytics

Batch Analytics provides a broader view of customer churn across an uploaded Telco customer dataset.

Users can investigate:

* Overall churn distribution
* High-risk customer populations
* Customer segments
* Churn-related characteristics
* Risk distributions
* Key customer metrics

This allows the platform to move from individual customer analysis toward portfolio-level customer intelligence.

---

# 7. Business Insights

The Business Insights section transforms model predictions and customer data into higher-level business information.

It is designed to help users understand:

* Overall customer risk
* Churn exposure
* High-risk customer segments
* Customer behavior patterns
* Potential retention opportunities
* Important churn drivers

This provides a bridge between Machine Learning output and business decision-making.

---

# 8. Interactive Web Application

The system is implemented as an interactive **Streamlit** application.

The interface provides a professional dark SaaS-style design with:

* CHURN INTELLIGENCE branding
* Premium dashboard layout
* Structured navigation
* Customer risk views
* Prediction interfaces
* Analytics dashboards
* Business insights
* Model/system status indicators
* About section
* Responsive data visualizations

---

# System Architecture

The overall system follows an end-to-end Machine Learning pipeline:

```text
                    TELCO CUSTOMER DATA
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Loading        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Machine Learning    │
                 │ Model               │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Churn Prediction    │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐   ┌─────────────────┐
        │ Risk Analysis   │   │ SHAP Explainable│
        │                 │   │ AI              │
        └────────┬────────┘   └────────┬────────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Retention           │
                 │ Recommendations     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Business Insights   │
                 │ & Analytics         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 └─────────────────────┘
```

---

# Machine Learning Pipeline

The Machine Learning workflow includes:

```text
Raw Telco Dataset
       ↓
Data Validation
       ↓
Data Cleaning
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
Train / Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Selection
       ↓
Prediction
       ↓
Explainability
       ↓
Business Recommendations
```

---

# Model Evaluation

Multiple classification metrics are used to evaluate model performance.

These include:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Because customer churn is a business-risk problem, **accuracy alone is not sufficient** for evaluating the effectiveness of a churn prediction model.

Particular attention is given to metrics such as:

* Recall
* Precision
* F1-Score
* ROC-AUC

These provide a more meaningful understanding of the model's ability to identify customers who may churn.

---

# Explainable AI Architecture

The explainability component is based on SHAP.

```text
Customer
   │
   ▼
Prediction Model
   │
   ▼
Churn Probability
   │
   ▼
SHAP Analysis
   │
   ├── Global Feature Importance
   │
   └── Individual Customer Explanation
                │
                ▼
        Main Risk Factors
                │
                ▼
        Retention Recommendation
```

This approach helps bridge the gap between Machine Learning predictions and human decision-making.

---

# Technology Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Core programming language      |
| Pandas       | Data manipulation and analysis |
| NumPy        | Numerical computing            |
| Scikit-learn | Machine Learning               |
| SHAP         | Explainable AI                 |
| Matplotlib   | Visualization                  |
| Seaborn      | Statistical visualization      |
| Plotly       | Interactive visualization      |
| Streamlit    | Web application and dashboard  |
| Git          | Version control                |
| GitHub       | Source code management         |

---

# Project Structure

```text
Customer-Churn-Prediction-System/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│
├── outputs/
│   ├── reports/
│   └── shap/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── explainability/
│   └── recommendation/
│
├── tests/
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd Customer-Churn-Prediction-System
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit application will open in your browser.

---

# Dataset

The project is based on the **Telco Customer Churn dataset**.

The dataset contains telecommunications customer information covering areas such as:

* Customer demographics
* Tenure
* Contract information
* Billing
* Monthly charges
* Total charges
* Phone services
* Internet services
* Additional services
* Payment methods
* Churn status

The prediction target is:

```text
Churn
```

where:

```text
Yes → Customer churned
No  → Customer remained
```

---

# Business Value

A churn prediction system can help telecommunications organizations move from reactive customer retention toward proactive customer management.

Instead of waiting for a customer to leave:

```text
Customer Behavior
       ↓
Risk Detection
       ↓
Churn Probability
       ↓
Explain Risk
       ↓
Recommend Action
       ↓
Retention Opportunity
```

Potential business benefits include:

* Earlier identification of high-risk customers
* Better customer segmentation
* More targeted retention campaigns
* Data-driven customer engagement
* Improved understanding of churn drivers
* More informed retention decisions

---

# Current Capabilities

The current version of CHURN INTELLIGENCE includes:

* [x] Telco customer churn prediction
* [x] Churn probability estimation
* [x] Customer risk classification
* [x] Individual customer risk analysis
* [x] SHAP Explainable AI
* [x] Global feature importance
* [x] Individual prediction explanations
* [x] Retention recommendations
* [x] Batch customer prediction
* [x] Batch analytics
* [x] Business insights
* [x] Interactive Streamlit dashboard
* [x] Premium dashboard UI
* [x] About section
* [x] Model/system status indicators
* [x] Telco-specific dataset validation

---

# Limitations

The current system has several important limitations.

### Industry Specificity

The trained model is designed for telecommunications customer churn.

It cannot be assumed to generalize to other industries without retraining and redesigning the relevant data pipeline.

### Dataset Dependency

Prediction quality depends on the quality, structure, and characteristics of the input customer data.

### Model-Based Predictions

Churn probabilities represent model estimates rather than guaranteed future customer behavior.

### Business Recommendations

Retention recommendations are decision-support suggestions and should be evaluated alongside business context, customer history, operational constraints, and company policies.

---

# Future Development

Possible future improvements include:

### Multi-Industry Support

Develop separate churn prediction pipelines for industries such as:

* SaaS
* Banking
* Insurance
* Retail
* Subscription services

Each industry would require its own dataset, feature engineering, model, explainability layer, and business logic.

### Advanced Model Optimization

Future versions could investigate:

* Hyperparameter optimization
* Ensemble approaches
* Advanced gradient boosting
* Calibration of churn probabilities
* Cost-sensitive learning
* Automated model selection

### MLOps

Potential future additions include:

* Model versioning
* Automated retraining
* Model monitoring
* Data drift detection
* Prediction monitoring
* Experiment tracking

### Deployment

The platform could eventually be deployed as a production cloud application with:

* Authentication
* Role-based access
* Secure data upload
* API-based predictions
* Database integration
* Monitoring and logging

---

# Project Vision

CHURN INTELLIGENCE is designed to evolve from a Machine Learning project into a broader **customer retention decision-support platform**.

The long-term concept is:

```text
                 CUSTOMER DATA
                      ↓
               DATA INTELLIGENCE
                      ↓
              CHURN PREDICTION
                      ↓
               RISK PROBABILITY
                      ↓
              EXPLAINABLE AI
                      ↓
          CUSTOMER-SPECIFIC INSIGHT
                      ↓
          RETENTION RECOMMENDATION
                      ↓
                BUSINESS ACTION
```

The core objective remains simple:

> **Predict churn. Explain the risk. Recommend an action.**

---

# Author

## Devinda Idamgedara

**BSc (Hons) Data Science Undergraduate**

Areas of interest:

* Data Science
* Machine Learning
* Artificial Intelligence
* Predictive Analytics
* Explainable AI
* Business Intelligence

---

# License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for more information.

---

## Final Note

CHURN INTELLIGENCE is an academic and portfolio-focused Machine Learning project demonstrating how predictive analytics, Explainable AI, and business intelligence can be combined into an end-to-end customer churn decision-support system.

The current implementation focuses specifically on the **telecommunications domain**.
