# 🔮 AI-Powered Customer Churn Prediction System

> **An end-to-end Machine Learning project for predicting customer churn, understanding why customers are at risk, and eventually recommending personalized retention strategies.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🌟 Project Overview

Customer churn is one of the major challenges faced by subscription-based businesses.

Acquiring a new customer is often more expensive than retaining an existing one. Therefore, being able to **identify customers who are likely to leave before they actually churn** can provide businesses with a valuable opportunity to take preventive action.

This project aims to build an **AI-powered Customer Churn Prediction System** that goes beyond simply predicting whether a customer will churn.

The final system is designed to:

* 🎯 Predict the probability of customer churn
* 🔍 Explain why a customer is considered high-risk
* 💡 Generate personalized retention recommendations
* 📊 Provide interactive business analytics
* 🌐 Offer a web interface for making predictions
* 📁 Allow businesses to upload customer datasets
* 🤖 Apply Explainable AI techniques such as **SHAP**

---

# 🎯 Project Objectives

The main objectives of this project are:

### 1. Predict Customer Churn

Develop Machine Learning models capable of predicting whether a customer is likely to leave the service.

### 2. Estimate Churn Probability

Instead of producing only a `Yes / No` prediction, the system will provide a probability score representing the customer's estimated churn risk.

### 3. Explain Predictions

Use Explainable AI techniques such as **SHAP (SHapley Additive exPlanations)** to identify the factors contributing to each prediction.

### 4. Recommend Retention Strategies

Use customer characteristics and model insights to suggest appropriate retention actions.

### 5. Visualize Customer Risk

Build interactive analytics that allow users to understand churn patterns and identify high-risk customer segments.

### 6. Provide a Usable Interface

Create a web-based interface where users can upload customer data and receive predictions and insights.

---

# 🧠 Planned System

The overall system is planned around the following pipeline:

```text
                 ┌──────────────────────┐
                 │   Customer Dataset   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Data Preprocessing │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Exploratory Data     │
                 │ Analysis (EDA)       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Machine Learning     │
                 │ Model Training       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Evaluation     │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐
        │ Churn           │    │ Explainable AI │
        │ Prediction      │    │ (SHAP)         │
        └────────┬────────┘    └────────┬────────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Retention Strategy   │
                 │ Recommendations      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Dashboard / Web App  │
                 └──────────────────────┘
```

---

# 📊 Dataset

The project uses the **Telco Customer Churn dataset**, which contains information about telecommunications customers and whether they eventually churned.

The dataset includes information related to:

* 👤 Customer demographics
* 📅 Customer tenure
* 📱 Phone services
* 🌐 Internet services
* 💳 Payment methods
* 💰 Monthly charges
* 💵 Total charges
* 📄 Contract type
* 🧾 Billing information
* 🚪 Customer churn status

The target variable is:

```text
Churn
```

where:

```text
Yes → Customer churned
No  → Customer stayed
```

---

# 🔬 Current Project Progress

The project is being developed step-by-step as an end-to-end Machine Learning system.

### ✅ Completed

* [x] Project environment setup
* [x] Dataset collection
* [x] Project folder structure
* [x] Initial dataset inspection
* [x] Exploratory Data Analysis
* [x] Churn distribution analysis
* [x] Categorical feature analysis
* [x] Numerical feature analysis
* [x] Churn rate visualization
* [x] Identification of important customer patterns


### 🚧 In Progress
* [ ] Data preprocessing
* [ ] Data type conversion
* [ ] Preparation of data for Machine Learning
* [ ] Feature engineering
* [ ] Train-test split
* [ ] Feature scaling
* [ ] Machine Learning model development
* [ ] Model comparison
* [ ] Hyperparameter tuning

### 🔮 Planned

* [ ] Model evaluation
* [ ] Churn probability prediction
* [ ] SHAP Explainable AI
* [ ] Personalized retention recommendations
* [ ] Interactive dashboard
* [ ] Customer data upload
* [ ] Web application
* [ ] Model deployment

---

# 📈 Exploratory Data Analysis

During the EDA phase, different customer attributes were investigated to understand their relationship with churn.

Some important patterns identified include:

### 📋 Contract Type

Customers with shorter-term contracts show a higher tendency to churn compared with customers on longer-term contracts.

### 🌐 Internet Service

Customers using **Fiber Optic** internet services showed a comparatively higher churn rate within the analyzed dataset.

### 💰 Monthly Charges

Higher monthly charges were associated with increased churn among the analyzed customers.

### 🛡️ Online Security & Technical Support

Customers without additional services such as online security and technical support showed higher churn tendencies.

These observations will later be used to support feature engineering and model interpretation.

---

# 🛠️ Technology Stack

| Technology                   | Purpose                           |
| ---------------------------- | --------------------------------- |
| 🐍 Python                    | Core programming language         |
| 🐼 Pandas                    | Data manipulation and analysis    |
| 🔢 NumPy                     | Numerical computing               |
| 📊 Matplotlib                | Data visualization                |
| 🎨 Seaborn                   | Statistical visualization         |
| 🤖 Scikit-learn              | Machine Learning                  |
| 🧠 SHAP                      | Explainable AI                    |
| 🌐 Streamlit / Web Framework | Planned application interface     |
| 📊 Plotly                    | Planned interactive visualization |
| 🐙 Git & GitHub              | Version control                   |

---

# 📁 Project Structure

```text
Customer-Churn-Prediction-System/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   │
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── visualization/
│   ├── models/
│   └── explainability/
│
├── app/
│
├── models/
│
├── reports/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── main.py
```

> **Note:** The project structure will evolve as new Machine Learning, explainability, dashboard, and deployment components are developed.

---

# 🧪 Machine Learning Strategy

The Machine Learning stage will investigate multiple classification algorithms rather than relying on a single model.

Potential models include:

```text
Logistic Regression
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
Other suitable classification models
```

Models will be compared using appropriate classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Because customer churn prediction is a business-risk problem, **accuracy alone will not be treated as the only measure of model performance.**

---

# 🔍 Explainable AI

A major goal of this project is to make Machine Learning predictions understandable.

Instead of simply saying:

```text
Customer → HIGH CHURN RISK
```

the final system aims to provide an explanation such as:

```text
⚠️ High Churn Risk

Main contributing factors:

• Month-to-month contract
• High monthly charges
• Fiber optic internet
• No online security
• Short customer tenure
```

SHAP will be investigated to understand both:

* 🌍 Global model behavior
* 👤 Individual customer predictions

---

# 💡 Personalized Retention Strategies

The final system will attempt to connect churn predictions with actionable business recommendations.

For example:

```text
High Churn Risk
        │
        ▼
Identify Risk Factors
        │
        ▼
Generate Customer-Specific Recommendation
        │
        ▼
Retention Action
```

Possible recommendations could include:

* 📑 Contract upgrade offers
* 💰 Personalized discounts
* 🛡️ Security service bundles
* 📞 Technical support assistance
* 🎁 Loyalty offers
* 📦 Service plan recommendations

---

# 📊 Planned Dashboard

The future dashboard will provide an interactive view of customer churn.

Possible dashboard components:

### 📌 Key Performance Indicators

* Total Customers
* Churned Customers
* Churn Rate
* Average Monthly Charges
* High-Risk Customers

### 📈 Visual Analytics

* Churn by Contract
* Churn by Internet Service
* Churn by Tenure
* Churn by Payment Method
* Churn by Monthly Charges
* Customer Risk Distribution

### 👤 Individual Customer Analysis

Users will eventually be able to inspect an individual customer's:

```text
Customer Profile
      ↓
Churn Probability
      ↓
Risk Level
      ↓
SHAP Explanation
      ↓
Recommended Retention Strategy
```

---

# 🚀 Future Vision

The long-term goal is to transform this project from a traditional Machine Learning model into a **decision-support system for customer retention**.

```text
Raw Customer Data
        ↓
Data Intelligence
        ↓
Churn Prediction
        ↓
Risk Probability
        ↓
Explainable AI
        ↓
Personalized Recommendation
        ↓
Business Action
```

The system should not only answer:

> **"Who is likely to churn?"**

but eventually:

> **"Why are they likely to churn, and what should the business do about it?"**

---

# 👨‍💻 Author

### Devinda Idamgedara

**BSc (Hons) Data Science Undergraduate**

Interested in:

`Data Science` • `Machine Learning` • `Artificial Intelligence` • `Predictive Analytics`

---

# 📜 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for more information.

---

⭐ **If you find this project interesting, consider giving the repository a star!**

> 🚧 **This project is actively under development.**
>
> New Machine Learning, Explainable AI, recommendation, visualization, and deployment features will be added progressively.
