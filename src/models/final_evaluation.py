import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Processed test data paths
X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "x_test_processed.csv"
Y_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Output directory
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Selected operating threshold
OPERATING_THRESHOLD = 0.30


def load_test_data():
    """Load the untouched processed test dataset."""

    x_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return x_test, y_test


def load_tuned_model():
    """Load the tuned Gradient Boosting model."""

    model_path = MODEL_DIR / "gradient_boosting_tuned.pkl"

    return joblib.load(model_path)


def evaluate_at_threshold(
    y_test,
    probabilities,
    threshold
):
    """Evaluate model predictions at a specific threshold."""

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": cm,
        "Predictions": y_pred
    }


def main():

    print("\nCustomer Churn - Final Model Evaluation")

    # Load test data
    x_test, y_test = load_test_data()

    print("\nTest data loaded successfully.")

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

    model = load_tuned_model()

    print(
        "Tuned Gradient Boosting model loaded successfully."
    )

    # Generate churn probabilities
    y_prob = model.predict_proba(
        x_test
    )[:, 1]

    # ROC-AUC does not depend on threshold
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    # Evaluate default threshold
    default_results = evaluate_at_threshold(
        y_test,
        y_prob,
        0.50
    )

    # Evaluate selected threshold
    selected_results = evaluate_at_threshold(
        y_test,
        y_prob,
        OPERATING_THRESHOLD
    )

    # Default threshold results
    print("\nDEFAULT THRESHOLD EVALUATION")

    print(
        f"\nThreshold : "
        f"{default_results['Threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{default_results['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{default_results['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{default_results['Recall']:.4f}"
    )

    print(
        f"F1-Score  : "
        f"{default_results['F1-Score']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(default_results["Confusion Matrix"])

    
    # Selected threshold results
    
    
    print("\nSELECTED THRESHOLD EVALUATION")
    

    print(
        f"\nThreshold : "
        f"{selected_results['Threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{selected_results['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{selected_results['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{selected_results['Recall']:.4f}"
    )

    print(
        f"F1-Score  : "
        f"{selected_results['F1-Score']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(selected_results["Confusion Matrix"])

    # Classification report
    
    print("\nCLASSIFICATION REPORT")
    

    print(
        classification_report(
            y_test,
            selected_results["Predictions"],
            target_names=[
                "No Churn",
                "Churn"
            ]
        )
    )

    
    # Comparison
    comparison = pd.DataFrame([
        {
            "Threshold": 0.50,
            "Accuracy": default_results["Accuracy"],
            "Precision": default_results["Precision"],
            "Recall": default_results["Recall"],
            "F1-Score": default_results["F1-Score"]
        },
        {
            "Threshold": OPERATING_THRESHOLD,
            "Accuracy": selected_results["Accuracy"],
            "Precision": selected_results["Precision"],
            "Recall": selected_results["Recall"],
            "F1-Score": selected_results["F1-Score"]
        }
    ])

    print("\nTHRESHOLD COMPARISON")
    

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Save final evaluation results
    results_path = (
        REPORT_DIR /
        "final_model_evaluation.csv"
    )

    comparison.to_csv(
        results_path,
        index=False
    )

    print(
        f"\nFinal evaluation results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()