import pandas as pd


# ============================================================
# Recommendation Rules
# ============================================================

RECOMMENDATION_RULES = {

    "New Customer": {
        "condition": lambda customer:
            customer.get("Tenure Months", 0) <= 12,

        "shap_features": [
            "num__Tenure Months",
            "cat__Tenure Group_New"
        ],

        "action_priority": "HIGH",

        "recommendation": (
            "Provide a new-customer retention offer such as "
            "onboarding assistance, a loyalty incentive, or "
            "a limited-time introductory benefit."
        ),

        "reason": (
            "The customer has short tenure, which is associated "
            "with elevated churn risk."
        )
    },

    "Month-to-month": {
        "condition": lambda customer:
            customer.get("Contract") == "Month-to-month",

        "shap_features": [
            "cat__Contract_Month-to-month"
        ],

        "action_priority": "HIGH",

        "recommendation": (
            "Offer a contract upgrade incentive to encourage "
            "the customer to move from a month-to-month plan "
            "to a longer-term contract."
        ),

        "reason": (
            "The customer's month-to-month contract is "
            "contributing positively to the predicted churn risk."
        )
    },

    "No Online Security": {
        "condition": lambda customer:
            customer.get("Online Security") == "No",

        "shap_features": [
            "cat__Online Security_No"
        ],

        "action_priority": "MEDIUM",

        "recommendation": (
            "Offer a discounted Online Security package or "
            "a limited-time security service promotion."
        ),

        "reason": (
            "The absence of Online Security is contributing "
            "to the customer's predicted churn risk."
        )
    },

    "No Tech Support": {
        "condition": lambda customer:
            customer.get("Tech Support") == "No",

        "shap_features": [
            "cat__Tech Support_No"
        ],

        "action_priority": "MEDIUM",

        "recommendation": (
            "Offer a technical support package or a "
            "limited-time technical support promotion."
        ),

        "reason": (
            "The absence of Tech Support is contributing "
            "to the customer's predicted churn risk."
        )
    },

    "High Monthly Charges": {
        "condition": lambda customer:
            customer.get("Monthly Charges", 0) >= 70,

        "shap_features": [
            "num__Monthly Charges"
        ],

        "action_priority": "MEDIUM",

        "recommendation": (
            "Review the customer's current plan and consider "
            "a personalized pricing offer or a lower-cost "
            "plan that preserves important services."
        ),

        "reason": (
            "The customer's monthly charges may be contributing "
            "to their predicted churn risk."
        )
    },

    "Electronic Check": {
        "condition": lambda customer:
            customer.get("Payment Method")
            == "Electronic check",

        "shap_features": [
            "cat__Payment Method_Electronic check"
        ],

        "action_priority": "LOW",

        "recommendation": (
            "Encourage the customer to switch to a more "
            "convenient automated payment method and "
            "consider offering a small billing incentive."
        ),

        "reason": (
            "The customer's current payment method is "
            "associated with increased churn risk."
        )
    },

    "Fiber Optic": {
        "condition": lambda customer:
            customer.get("Internet Service")
            == "Fiber optic",

        "shap_features": [
            "cat__Internet Service_Fiber optic"
        ],

        "action_priority": "LOW",

        "recommendation": (
            "Review the customer's fiber optic plan and "
            "ensure the customer is receiving sufficient "
            "value, performance, and support for the price."
        ),

        "reason": (
            "Fiber optic service is contributing positively "
            "to this customer's predicted churn risk."
        )
    },

    "No Online Backup": {
        "condition": lambda customer:
            customer.get("Online Backup") == "No",

        "shap_features": [
            "cat__Online Backup_No"
        ],

        "action_priority": "LOW",

        "recommendation": (
            "Offer an Online Backup package if additional "
            "digital services could improve customer value."
        ),

        "reason": (
            "The customer does not currently use Online Backup."
        )
    },

    "No Device Protection": {
        "condition": lambda customer:
            customer.get("Device Protection") == "No",

        "shap_features": [
            "cat__Device Protection_No"
        ],

        "action_priority": "LOW",

        "recommendation": (
            "Offer a Device Protection package as an "
            "optional value-added service."
        ),

        "reason": (
            "The customer does not currently use Device Protection."
        )
    }
}


# ============================================================
# SHAP Impact
# ============================================================

def get_shap_impact(
    rule,
    shap_explanation
):
    """
    Find the strongest positive SHAP impact associated
    with a recommendation rule.
    """

    if shap_explanation is None:
        return None

    if not isinstance(
        shap_explanation,
        pd.DataFrame
    ):
        return None

    shap_features = rule.get(
        "shap_features",
        []
    )

    matching_rows = shap_explanation[
        shap_explanation["Feature"].isin(
            shap_features
        )
    ]

    if matching_rows.empty:
        return None

    # Only consider factors that increase churn.
    positive_rows = matching_rows[
        matching_rows["SHAP Value"] > 0
    ]

    if positive_rows.empty:
        return None

    strongest_row = positive_rows.loc[
        positive_rows["SHAP Value"].idxmax()
    ]

    return {
        "feature": strongest_row["Feature"],
        "value": float(
            strongest_row["SHAP Value"]
        )
    }


# ============================================================
# Model Impact Level
# ============================================================

def determine_model_impact(
    shap_value
):
    """Convert SHAP magnitude into an impact level."""

    if shap_value >= 0.40:
        return "HIGH"

    elif shap_value >= 0.20:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# Generate Recommendations
# ============================================================

def generate_recommendations(
    customer,
    prediction_result
):
    """
    Generate personalized retention recommendations
    based on customer attributes and positive SHAP factors.
    """

    recommendations = []

    churn_probability = prediction_result[
        "churn_probability"
    ]

    risk_level = prediction_result[
        "risk_level"
    ]

    shap_explanation = prediction_result.get(
        "shap_explanation"
    )

    # --------------------------------------------------------
    # Do not generate retention actions for low-risk customers
    # --------------------------------------------------------

    if risk_level == "LOW":

        return {
            "churn_probability": churn_probability,
            "risk_level": risk_level,
            "recommendation_count": 0,
            "recommendations": []
        }

    # --------------------------------------------------------
    # Evaluate rules
    # --------------------------------------------------------

    for rule_name, rule in RECOMMENDATION_RULES.items():

        # Check raw customer condition
        try:
            condition_met = rule["condition"](
                customer
            )

        except Exception:
            condition_met = False

        if not condition_met:
            continue

        # Find positive SHAP contribution
        shap_result = get_shap_impact(
            rule,
            shap_explanation
        )

        # Only recommend if SHAP confirms
        # that the factor increases churn.
        if shap_result is None:
            continue

        shap_value = shap_result["value"]

        model_impact = determine_model_impact(
            shap_value
        )

        recommendations.append({
            "trigger": rule_name,

            "action_priority": rule[
                "action_priority"
            ],

            "model_impact": model_impact,

            "shap_feature": shap_result[
                "feature"
            ],

            "shap_impact": shap_value,

            "reason": rule[
                "reason"
            ],

            "recommendation": rule[
                "recommendation"
            ]
        })

    # --------------------------------------------------------
    # Sort by SHAP impact
    # --------------------------------------------------------

    recommendations.sort(
        key=lambda item:
            item["shap_impact"],
        reverse=True
    )

    # --------------------------------------------------------
    # Limit recommendations
    # --------------------------------------------------------

    recommendations = recommendations[:5]

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        recommendation["rank"] = index

    return {
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "recommendation_count": len(
            recommendations
        ),
        "recommendations": recommendations
    }


# ============================================================
# Display Recommendations
# ============================================================

def display_recommendations(
    recommendation_result
):
    """Display personalized recommendations."""

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PERSONALIZED RETENTION RECOMMENDATIONS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nChurn Probability: "
        f"{recommendation_result['churn_probability']:.2%}"
    )

    print(
        f"Risk Level: "
        f"{recommendation_result['risk_level']}"
    )

    print(
        f"Recommendations: "
        f"{recommendation_result['recommendation_count']}"
    )

    if (
        recommendation_result[
            "recommendation_count"
        ] == 0
    ):

        print(
            "\nNo immediate retention action required."
        )

        return

    print(
        "\nPriority Retention Actions:"
    )

    for recommendation in (
        recommendation_result[
            "recommendations"
        ]
    ):

        print(
            f"\n{recommendation['rank']}. "
            f"{recommendation['trigger']}"
        )

        print(
            f"   Model Impact: "
            f"{recommendation['model_impact']}"
        )

        print(
            f"   SHAP Impact: "
            f"{recommendation['shap_impact']:+.4f}"
        )

        print(
            f"   Action Priority: "
            f"{recommendation['action_priority']}"
        )

        print(
            f"   Reason: "
            f"{recommendation['reason']}"
        )

        print(
            f"   Action: "
            f"{recommendation['recommendation']}"
        )

    print(
        "\n"
        + "=" * 75
    )


# ============================================================
# Test
# ============================================================

def main():

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

    # Example SHAP explanation from prediction system
    shap_explanation = pd.DataFrame({
        "Feature": [
            "num__Tenure Months",
            "cat__Contract_Month-to-month",
            "cat__Internet Service_Fiber optic",
            "cat__Online Security_No",
            "cat__Tech Support_No",
            "cat__Multiple Lines_No"
        ],
        "SHAP Value": [
            0.619629,
            0.460298,
            0.277981,
            0.249192,
            0.234217,
            -0.179773
        ]
    })

    prediction_result = {
        "churn_probability": 0.7756,
        "prediction": 1,
        "prediction_label": "Churn",
        "risk_level": "HIGH",
        "shap_explanation": shap_explanation
    }

    result = generate_recommendations(
        customer,
        prediction_result
    )

    display_recommendations(
        result
    )


if __name__ == "__main__":
    main()