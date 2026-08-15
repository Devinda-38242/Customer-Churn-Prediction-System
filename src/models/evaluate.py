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


def load_test_data():
    """Load the processed test dataset."""

    x_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return x_test, y_test


def load_model(model_filename):
    """Load a trained model."""

    model_path = MODEL_DIR / model_filename

    return joblib.load(model_path)


def evaluate_model(model, x_test, y_test, model_name):
    """Evaluate a classification model."""

    # Generate predictions
    y_pred = model.predict(x_test)

    # Generate churn probabilities
    y_prob = model.predict_proba(x_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name.upper()} - Model Evaluation")

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix: ")
    print(cm)

    print("\nClassification Report: ")
    print(classification_report(y_test, y_pred))

    # Return metrics
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }


def main():

    print("\nCustomer Churn - Model comparison")

    # Load test data
    x_test, y_test = load_test_data()

    print("\nTest data loaded successfully.")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Models to evaluate
    models = {
        "Logistic Regression": "logistic_regression.pkl",
        "Tuned Logistic Regression": "logistic_regression_tuned.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Tuned Gradient Boosting": "gradient_boosting_tuned.pkl"
    }

    # Store evaluation results
    results = []

    # Evaluate every model
    for model_name, model_filename in models.items():

        print(f"\nLoading {model_name}...")

        model = load_model(model_filename)

        result = evaluate_model(
            model,
            x_test,
            y_test,
            model_name
        )

        results.append(result)

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)

    # Sort by ROC-AUC
    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\nModel Comparison")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Save comparison results
    results_path = REPORT_DIR / "baseline_model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print(f"\nComparison results saved to: {results_path}")


if __name__ == "__main__":
    main()