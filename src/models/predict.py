import sys
from pathlib import Path

import pandas as pd
import joblib
import shap

from src.data.feature_engineering import add_features



# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

# Import Feature Engineering
from src.data.feature_engineering import add_features


# Model / Preprocessor Paths
MODEL_PATH = MODEL_DIR / "gradient_boosting_tuned.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"



# Prediction Configuration
THRESHOLD = 0.30

HIGH_RISK_THRESHOLD = 0.70



# Required Raw Customer Columns
REQUIRED_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges"
]



# Load Model
def load_model():
    """Load the tuned Gradient Boosting model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model



# Load Preprocessor
def load_preprocessor():
    """Load the fitted preprocessing pipeline."""

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return preprocessor



# Validate Customer Data
def validate_customer_data(customer_data):
    """
    Validate raw customer data.

    Accepts either:
        - pandas DataFrame
        - dictionary
    """

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    elif not isinstance(customer_data, pd.DataFrame):
        raise TypeError(
            "customer_data must be a pandas DataFrame "
            "or dictionary."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in customer_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required customer columns: "
            + ", ".join(missing_columns)
        )

    return customer_data.copy()

# Prepare Customer Data

def prepare_customer_data(customer_data):
    """
    Clean and feature-engineer raw customer data
    using the same logic used during training.
    """

    customer_data = validate_customer_data(
        customer_data
    )

    # Convert Total Charges to numeric
    customer_data["Total Charges"] = pd.to_numeric(
        customer_data["Total Charges"],
        errors="coerce"
    )

    # Check for invalid Total Charges
    if customer_data["Total Charges"].isnull().any():
        raise ValueError(
            "Total Charges contains invalid or missing values."
        )

    # Apply the exact feature engineering used during training
    customer_data = add_features(
        customer_data
    )

    return customer_data

# Preprocess Customer Data

def preprocess_customer_data(
    customer_data,
    preprocessor
):
    """
    Apply the fitted preprocessing pipeline.
    """

    processed_data = preprocessor.transform(
        customer_data
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    processed_df = pd.DataFrame(
        processed_data,
        columns=feature_names,
        index=customer_data.index
    )

    return processed_df



# Determine Risk Level
def determine_risk_level(
    churn_probability
):
    """Convert churn probability into a risk level."""

    if churn_probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    elif churn_probability >= THRESHOLD:
        return "MEDIUM"

    else:
        return "LOW"



# Generate SHAP Explanation

def generate_shap_explanation(
    model,
    processed_customer
):
    """
    Generate SHAP explanation for one customer.
    """

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        processed_customer
    )

    # Select first customer
    customer_shap_values = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": processed_customer.columns,
        "Feature Value": processed_customer.iloc[0].values,
        "SHAP Value": customer_shap_values
    })

    # Remove inactive one-hot encoded features
    categorical_mask = (
        explanation["Feature"]
        .str.startswith("cat__")
    )

    inactive_categorical = (
        categorical_mask
        & (explanation["Feature Value"] == 0)
    )

    explanation = explanation[
        ~inactive_categorical
    ].copy()

    # Factors increasing churn
    positive_factors = (
        explanation[
            explanation["SHAP Value"] > 0
        ]
        .sort_values(
            by="SHAP Value",
            ascending=False
        )
        .head(5)
        .reset_index(drop=True)
    )

    # Factors decreasing churn
    negative_factors = (
        explanation[
            explanation["SHAP Value"] < 0
        ]
        .sort_values(
            by="SHAP Value",
            ascending=True
        )
        .head(5)
        .reset_index(drop=True)
    )

    return {
        "all_factors": explanation,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors
    }



# Main Prediction Function
def predict_customer(customer_data):
    """
    Predict customer churn using the tuned
    Gradient Boosting model.

    Parameters
    ----------
    customer_data : dict or pandas.DataFrame
        Raw customer information.

    Returns
    -------
    dict
        Churn probability, prediction, risk level,
        and SHAP explanation.
    """

    print("\nCustomer Churn Prediction")

    
    # Load artifacts

    print("\nLoading tuned Gradient Boosting model...")

    model = load_model()

    print("Model loaded successfully.")

    print("\nLoading preprocessing pipeline...")

    preprocessor = load_preprocessor()

    print("Preprocessor loaded successfully.")

    
    # Prepare raw customer data


    print("\nPreparing customer data...")

    prepared_customer = prepare_customer_data(
        customer_data
    )

    
    # Apply preprocessing
    

    print("Applying preprocessing...")

    processed_customer = preprocess_customer_data(
        prepared_customer,
        preprocessor
    )

    print(
        f"Processed customer shape: "
        f"{processed_customer.shape}"
    )

    
    # Generate churn probability
    

    churn_probability = model.predict_proba(
        processed_customer
    )[0, 1]

    
    # Apply threshold
   

    prediction = int(
        churn_probability >= THRESHOLD
    )

    
    # Determine risk level
    

    risk_level = determine_risk_level(
        churn_probability
    )

   
    # Generate SHAP explanation
    

    print("\nGenerating SHAP explanation...")

    shap_result = generate_shap_explanation(
        model,
        processed_customer
    )

    
    # Final result
  

    result = {
        "churn_probability": float(
            churn_probability
        ),
        "prediction": prediction,
        "prediction_label": (
            "Churn"
            if prediction == 1
            else "No Churn"
        ),
        "risk_level": risk_level,
        "threshold": THRESHOLD,
        "positive_factors": (
            shap_result["positive_factors"]
        ),
        "negative_factors": (
            shap_result["negative_factors"]
        ),
        "shap_explanation": (
            shap_result["all_factors"]
        )
    }

    return result



# Display Prediction


def display_prediction(result):
    """Display prediction results."""


    print(
        "\nCUSTOMER CHURN PREDICTION"
    )

   
    print(
        f"\nChurn Probability : "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Prediction        : "
        f"{result['prediction_label']}"
    )

    print(
        f"Risk Level        : "
        f"{result['risk_level']}"
    )

    print(
        f"Decision Threshold: "
        f"{result['threshold']:.2f}"
    )

    print(
        "\nTop Factors Increasing Churn:"
    )

    print(
        result["positive_factors"][
            ["Feature", "SHAP Value"]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTop Factors Reducing Churn:"
    )

    print(
        result["negative_factors"][
            ["Feature", "SHAP Value"]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 55
    )


# Test Prediction

def main():

    print(
        "\nCustomer Churn - Prediction System"
    )

    # Example customer
    customer = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 3,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.50,
        "Total Charges": 256.50
    }

    result = predict_customer(
        customer
    )

    display_prediction(
        result
    )


if __name__ == "__main__":
    main()