import pandas as pd


def clean_feature_name(feature):
    """Convert encoded feature names into human-readable names."""

    if feature.startswith("num__"):
        feature = feature.replace("num__", "")

    elif feature.startswith("cat__"):
        feature = feature.replace("cat__", "")

    # Convert one-hot encoded feature names
    replacements = {
        "_Month-to-month": ": Month-to-month",
        "_One year": ": One year",
        "_Two year": ": Two year",

        "_Yes": ": Yes",
        "_No": ": No",

        "New Customer + Month-to-Month":
            "New Customer + Month-to-Month"
    }

    for old, new in replacements.items():
        if old in feature:
            feature = feature.replace(old, new)

    return feature


def get_feature_direction(feature, feature_value, shap_value):
    """
    Determine whether a feature increases or decreases
    churn risk.
    """

    if shap_value > 0:
        return "Increases churn risk"

    return "Reduces churn risk"


def format_shap_explanation(
    positive_factors,
    negative_factors,
    top_n=5
):
    """Create human-readable SHAP explanations."""

    positive_results = []
    negative_results = []

    # Factors increasing churn
    for _, row in positive_factors.head(top_n).iterrows():

        feature = clean_feature_name(
            row["Feature"]
        )

        positive_results.append({
            "Feature": feature,
            "Feature Value": row.get(
                "Feature Value",
                None
            ),
            "SHAP Value": row["SHAP Value"],
            "Impact": "Increases churn risk"
        })

    # Factors reducing churn
    for _, row in negative_factors.head(top_n).iterrows():

        feature = clean_feature_name(
            row["Feature"]
        )

        negative_results.append({
            "Feature": feature,
            "Feature Value": row.get(
                "Feature Value",
                None
            ),
            "SHAP Value": row["SHAP Value"],
            "Impact": "Reduces churn risk"
        })

    return (
        pd.DataFrame(positive_results),
        pd.DataFrame(negative_results)
    )


def print_human_readable_explanation(
    churn_probability,
    prediction,
    risk_level,
    positive_results,
    negative_results
):
    """Print a user-friendly customer explanation."""

    
    print("\nCUSTOMER CHURN RISK EXPLANATION")
    

    print(
        f"\nChurn Probability : "
        f"{churn_probability * 100:.2f}%"
    )

    print(
        f"Prediction        : "
        f"{'Churn' if prediction == 1 else 'No Churn'}"
    )

    print(
        f"Risk Level        : "
        f"{risk_level}"
    )

    print(
        "\nFactors Increasing Churn Risk"
    )
    print("-" * 60)

    if positive_results.empty:
        print("No significant risk-increasing factors found.")

    else:

        for _, row in positive_results.iterrows():

            print(
                f"• {row['Feature']}"
                f"  "
                f"(SHAP: {row['SHAP Value']:.3f})"
            )

    print(
        "\nFactors Reducing Churn Risk"
    )
    print("-" * 60)

    if negative_results.empty:
        print("No significant risk-reducing factors found.")

    else:

        for _, row in negative_results.iterrows():

            print(
                f"• {row['Feature']}"
                f"  "
                f"(SHAP: {row['SHAP Value']:.3f})"
            )


if __name__ == "__main__":

    positive = pd.DataFrame({
        "Feature": [
            "cat__Online Security_No",
            "cat__Streaming TV_Yes",
            "cat__Partner_Yes"
        ],
        "SHAP Value": [
            0.171,
            0.085,
            0.080
        ]
    })

    negative = pd.DataFrame({
        "Feature": [
            "cat__Contract_Two year",
            "num__Tenure Months",
            "cat__Tech Support_No"
        ],
        "SHAP Value": [
            -0.407,
            -0.412,
            -0.218
        ]
    })

    positive_results, negative_results = (
        format_shap_explanation(
            positive,
            negative
        )
    )

    print_human_readable_explanation(
        churn_probability=0.0421,
        prediction=0,
        risk_level="LOW",
        positive_results=positive_results,
        negative_results=negative_results
    )