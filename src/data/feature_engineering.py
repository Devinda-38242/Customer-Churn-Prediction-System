import pandas as pd

def add_features(df):
    """
    Add business-focused features to the customer churn dataset.

    Features created:
    1. Additional Services Count
    2. Tenure Group
    3. New Customer + Month-to-Month flag

    Parameters:
        df (pd.DataFrame): Cleaned customer dataframe

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """

    # Work ona copy to avoid modifying the original dataframe.
    df = df.copy()

    # Additional Services Count
    service_columns = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies"
    ]

    # Count only services where the customer has "yes"
    df["Additional Services Count"] = (
        df[service_columns] == "Yes"
    ).sum(axis = 1)

    # Tenure Groups
    def assign_tenure_group(tenure):
        if tenure <= 12:
            return "New"
        elif tenure <= 24:
            return "Early"
        elif tenure <= 48:
            return "Established"
        else:
            return "Long-term"

    df["Tenure Group"] = df["Tenure Months"].apply(assign_tenure_group)


    

    # New Customer + Month to Month
    new_customer = df["Tenure Months"] <= 12
    month_to_month = df["Contract"] == "Month-to-month"

    df["New Customer + Month-to-Month"] = (
        new_customer & month_to_month
    ).astype(int)

    return df