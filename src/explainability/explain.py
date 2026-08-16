import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from explanation_formatter import (
    format_shap_explanation,
    print_human_readable_explanation
)


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Processed data paths
X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "x_test_processed.csv"
)

Y_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "y_test.csv"
)

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Output directories
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SHAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_test_data():
    """Load the processed test dataset."""

    x_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return x_test, y_test


def load_model():
    """Load the tuned Gradient Boosting model."""

    model_path = (
        MODEL_DIR
        / "gradient_boosting_tuned.pkl"
    )

    return joblib.load(model_path)


def calculate_shap_values(model, x_test):
    """Calculate SHAP values using TreeExplainer."""

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        x_test
    )

    return explainer, shap_values


def create_feature_importance(
    x_test,
    shap_values
):
    """Create global SHAP feature importance."""

    # Mean absolute SHAP value
    mean_abs_shap = (
        abs(shap_values)
        .mean(axis=0)
    )

    feature_importance = pd.DataFrame({
        "Feature": x_test.columns,
        "Mean Absolute SHAP": mean_abs_shap
    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="Mean Absolute SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return feature_importance


def save_feature_importance(
    feature_importance
):
    """Save SHAP feature importance report."""

    output_path = (
        REPORT_DIR
        / "shap_feature_importance.csv"
    )

    feature_importance.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSHAP feature importance saved to:"
        f"\n{output_path}"
    )


def create_summary_plot(
    shap_values,
    x_test
):
    """Create and save SHAP summary plot."""

    print("\nCreating SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        x_test,
        show=False
    )

    plt.tight_layout()

    output_path = (
        SHAP_DIR
        / "shap_summary_plot.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    print(
        f"SHAP summary plot saved to:"
        f"\n{output_path}"
    )


def explain_customer(
    model,
    explainer,
    x_test,
    customer_index,
    threshold=0.30
):
    """Explain the churn prediction for one customer."""

    print(
        f"\nIndividual Customer Explanation"
    )

    print(
        f"Customer index: {customer_index}"
    )

    # Select one customer
    customer = x_test.iloc[[customer_index]]

    # Generate churn probability
    churn_probability = model.predict_proba(
        customer
    )[0, 1]

    # Apply selected operating threshold
    prediction = int(
        churn_probability >= threshold
    )

    # Risk level
    if churn_probability >= 0.70:
        risk_level = "HIGH"
    elif churn_probability >= threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Calculate SHAP values
    customer_shap_values = explainer.shap_values(
        customer
    )[0]

    # Create explanation DataFrame
    explanation = pd.DataFrame({
        "Feature": customer.columns,
        "Feature Value": customer.iloc[0].values,
        "SHAP Value": customer_shap_values
    })

    # Remove inactive one-hot encoded features
    categorical_mask = explanation["Feature"].str.startswith("cat__")

    inactive_categorical = (
        categorical_mask
        & (explanation["Feature Value"] == 0)
    )

    explanation = explanation[
        ~inactive_categorical
    ].copy()

    # Positive SHAP values increase churn prediction
    positive_factors = (
        explanation[
            explanation["SHAP Value"] > 0
        ]
        .sort_values(
            by="SHAP Value",
            ascending=False
        )
    )

    # Negative SHAP values decrease churn prediction
    negative_factors = (
        explanation[
            explanation["SHAP Value"] < 0
        ]
        .sort_values(
            by="SHAP Value",
            ascending=True
        )
    )

        # Format SHAP explanations
    positive_results, negative_results = (
        format_shap_explanation(
            positive_factors,
            negative_factors,
            top_n=5
        )
    )

    # Print human-readable explanation
    print_human_readable_explanation(
        churn_probability=churn_probability,
        prediction=prediction,
        risk_level=risk_level,
        positive_results=positive_results,
        negative_results=negative_results
    )

    


    return {
        "probability": churn_probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "explanation": explanation,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors
    }

def create_waterfall_plot(
    explainer,
    customer,
    customer_shap_values,
    customer_index
):
    """Create a SHAP waterfall plot for one customer."""

    base_value = explainer.expected_value

    # Handle numpy arrays
    if hasattr(base_value, "__len__"):
        base_value = base_value[0]

    explanation = shap.Explanation(
        values=customer_shap_values,
        base_values=base_value,
        data=customer.iloc[0].values,
        feature_names=customer.columns
    )

    print(
        "\nCreating individual SHAP waterfall plot..."
    )

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    output_path = (
        SHAP_DIR
        / f"customer_{customer_index}_waterfall.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    print(
        f"Individual SHAP plot saved to:"
        f"\n{output_path}"
    )


def main():

    print(
        "\nCustomer Churn - SHAP Explainability"
    )

    # Load test data
    x_test, y_test = load_test_data()

    print(
        "\nTest data loaded successfully."
    )

    print(
        f"x_test shape: {x_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    # Load tuned model
    print(
        "\nLoading tuned Gradient Boosting model..."
    )

    model = load_model()

    print(
        "Tuned Gradient Boosting model loaded successfully."
    )

    # Calculate SHAP values
    explainer, shap_values = (
        calculate_shap_values(
            model,
            x_test
        )
    )

    print("\nSHAP values calculated successfully.")

    print(
        f"SHAP values shape: "
        f"{shap_values.shape}"
    )

    # Create feature importance
    feature_importance = (
        create_feature_importance(
            x_test,
            shap_values
        )
    )

    # Display top features
    print(
        "\nTop 15 Features by "
        "Mean Absolute SHAP Value"
    )

    print(
        feature_importance
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # Save feature importance
    save_feature_importance(
        feature_importance
    )

    # Create summary plot
    create_summary_plot(
        shap_values,
        x_test
    )

        
    # Individual customer explanation
    customer_index = 0

    customer = x_test.iloc[[customer_index]]

    customer_shap_values = explainer.shap_values(
        customer
    )[0]

    explanation_result = explain_customer(
        model=model,
        explainer=explainer,
        x_test=x_test,
        customer_index=customer_index,
        threshold=0.30
    )

    # Create individual SHAP waterfall plot
    create_waterfall_plot(
        explainer=explainer,
        customer=customer,
        customer_shap_values=customer_shap_values,
        customer_index=customer_index
    )

    print(
        "\nSHAP global analysis completed successfully."
    )






if __name__ == "__main__":
    main()