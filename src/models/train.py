import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Project Root directoy
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Processed data paths
X_TRAIN_PATH = PROJECT_ROOT/"data"/"processed"/"x_train_processed.csv"
X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "x_test_processed.csv"
Y_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "y_train.csv"
Y_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

# Model output directory
MODEL_DIR = PROJECT_ROOT/"models"
MODEL_DIR.mkdir(parents = True, exist_ok = True)

def load_processed_data():
    """Load the preprocessed training and testing datasets"""

    x_train = pd.read_csv(X_TRAIN_PATH)
    x_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return x_train,x_test,y_train,y_test

def train_logistic_regression(x_train, y_train):
    """Train the baseline model"""

    model = LogisticRegression(
        max_iter = 1000,
        random_state = 42
    )

    model.fit(x_train,y_train)

    return model

def train_decision_tree(x_train, y_train):
    """Train the Decision Tree model"""

    model = DecisionTreeClassifier(
        random_state = 42
    )

    model.fit(x_train, y_train)

    return model

def train_random_forest(x_train, y_train):
    """Train the Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(x_train, y_train)

    return model

def train_gradient_boosting(x_train, y_train):
    """Train the Gradient Boosting model"""

    model = GradientBoostingClassifier(
        n_estimators = 100,
        random_state = 42
    )

    model.fit(x_train, y_train)

    return model


def main():

    print("\nCustomer Churn - Model Developement")

    # Load processed data
    x_train, x_test, y_train, y_test = load_processed_data()

    print("\nProcessed data loaded successfully.")
    print(f"x_train shape: {x_train.shape} ")
    print(f"x_test shape: {x_test.shape} ")
    print(f"y_train shape: {y_train.shape} ")
    print(f"y_test shape: {y_test.shape} ")

    # Train baseline model
    print("\nTraining Logistic Regression...")

    model = train_logistic_regression(x_train, y_train)

    print("\nLogitic regression trainning completed.")

    # Save logistic regression model
    model_path = MODEL_DIR / "logistic_regression.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")

    # Train Decision tree
    print("\nTraining Decision Tree..")

    decision_tree = train_decision_tree(x_train, y_train)

    print("\nDecision Tree training completed.")

    # Save decision tree model
    decision_tree_path = MODEL_DIR / "decision_tree.pkl"

    joblib.dump(decision_tree, decision_tree_path)

    print(f"\nDecision Tree model saved to: {decision_tree_path}")

    # Train Random Forest
    print("\nTraining Random Forest...")

    random_forest = train_random_forest(x_train, y_train)

    print("\nRandom Forest training completed.")

    # Save Random Forest model.
    random_forest_path = MODEL_DIR / "random_forest.pkl"

    joblib.dump(random_forest, random_forest_path)

    print(f"Random Forest model saved to: {random_forest_path}")

    # Train Gradient Boosting
    print("\nTraining gradient Boosting...")

    gradient_boosting = train_gradient_boosting(x_train, y_train)

    print("\nGradient Boosting training completed.")

    # Save Gradient Boosting model.
    gradient_boosting_path = MODEL_DIR / "gradient_boosting.pkl"

    joblib.dump(gradient_boosting, gradient_boosting_path)

    print(f"Gradient Boosting model saved to: "
        f"{gradient_boosting_path}")

if __name__ == "__main__":
    main()