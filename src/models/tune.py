import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Processed data paths
X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "x_train_processed.csv"
Y_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "y_train.csv"

# Model output directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    """Load the processed training dataset."""

    x_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    return x_train, y_train

# Tune Logistic Regression using GridSearchCV.
def tune_logistic_regression(x_train, y_train):

    model = LogisticRegression(
        max_iter = 1000,
        random_state = 42
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring = "roc_auc",
        cv = 5,
        n_jobs = -1,
        verbose = 1
    )

    grid_search.fit(x_train, y_train)

    return grid_search

# Tune gradient Boost
def tune_gradient_boosting(x_train, y_train):

    model = GradientBoostingClassifier(
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    return grid_search

def main():

    print("\nCustomer Churn Logistic Regression Tuning")

    # Load training data
    x_train, y_train = load_training_data()

    print("\nTraining data loaded successfully.")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Perform hyperparameter tuning.
    print("\nStarting GridSearchCV...")

    grid_search = tune_logistic_regression(
        x_train,
        y_train
    )

    # Display best parameters
    print("\nBest Parameters")

    print(grid_search.best_params_)

    # Display best cross-validation score
    print("\nBest Cross-Validattion ROC-AUC: ")
    print(f"{grid_search.best_score_: 4f}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Save tuned model
    model_path = MODEL_DIR / "logistic_regression_tuned.pkl"

    joblib.dump(best_model, model_path)

    print(f"\nTuned model saved to: {model_path}")

    # Tune Gradient Boosting
    print("\nGRADIENT BOOSTING HYPERPARAMETER TUNING")

    print("\nStarting GridSearchCV...")

    gradient_search = tune_gradient_boosting(
        x_train,
        y_train
    )

    print("\nBest Parameters:")
    print(gradient_search.best_params_)

    print("\nBest Cross-Validation ROC-AUC:")
    print(f"{gradient_search.best_score_:.6f}")

    # Save tuned Gradient Boosting model
    gradient_model_path = MODEL_DIR / "gradient_boosting_tuned.pkl"

    joblib.dump(
        gradient_search.best_estimator_,
        gradient_model_path
    )

    print(
        f"\nTuned Gradient Boosting model saved to: "
        f"{gradient_model_path}"
    )


if __name__ == "__main__":
    main()