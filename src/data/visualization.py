import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

# Load the data set
df = load_data()

# Display the first few rows of the data set
print("First Few Rows: ")
print(df.head())

# Check the data set shape
print("\nDataset Shape: ")
print(df.shape)

# Display column names
print("\nColumn Names: ")
print(df.columns.tolist())

# Display data set information
print("\nDataset Information: ")
print(df.info())

# Check for missing values
print("\nMissing Values: ")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows: ")
print(df.duplicated().sum())

# Display Numerical Summary Statistics
print("\nNumerical Summary Statistics: ")
print(df.describe())

# Display Categorical Summary Statistics
print("\nCategorical Summary Statistics: ")
print(df.describe(include="object"))

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# Visualize the target variable
plt.figure()

sns.countplot(
    data = df,
    x = "Churn Label"
)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()

# Display Churn counts
print("\nChurn Distribution: ")
print(df["Churn Label"].value_counts())

# Display Churn Percentages
print("\nChurn Percentages: ")
print(df["Churn Label"].value_counts(normalize=True) * 100)

# Visualize the relationship between Contract type and Churn
plt.figure(figsize = (10,6))

sns.countplot(
    data = df,
    x = "Contract",
    hue = "Churn Label"
)
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.legend(title = "Churn")

plt.tight_layout()
plt.show()

# Visualize the relationship between Tenure Months and Churn
plt.figure(figsize = (10,6))

sns.boxplot(
    data = df,
    x = "Churn Label",
    y = "Tenure Months"
)

plt.title("Customer Churn by Tenure Months")
plt.xlabel("Churn Label")
plt.ylabel("Tenure Months")

plt.tight_layout()
plt.show()

# Visualize the relationship between Monthly Charges and Churn
plt.figure(figsize = (10,6))

sns.boxplot(
    data = df,
    x = "Churn Label",
    y = "Monthly Charges"
)

plt.title("Customer Churn by Monthly Charges")
plt.xlabel("Churn Label")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()

# Visualize the relationship between Internet Service and Churn
plt.figure(figsize = (10,6))

sns.countplot(
    data = df,
    x = "Internet Service",
    hue = "Churn Label"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.legend(title = "Churn")

plt.tight_layout()
plt.show()

# Calculate the Churn Rate by Internet Service
internet_churn_rate = (
    df.groupby("Internet Service")["Churn Value"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nChurn Rate by Internet Service: ")
print(internet_churn_rate)

# Visualize the Churn Rate by Internet Service
plt.figure(figsize = (10,6))

sns.barplot(
    x = internet_churn_rate.index,
    y = internet_churn_rate.values
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")

# Display percentage values on top of the bars
for i, value in enumerate(internet_churn_rate.values):
    plt.text(
        i,
    value + 1,  # Position the text slightly above the bar
    f'{value:.2f}%',  # Format the value as a percentage with two decimal places
    ha='center'
)

plt.ylim(0,50) # Set y-axis limit to 50% for better visualization
plt.tight_layout()
plt.show()

# Visualize the relationship between Payment Method and Churn
plt.figure(figsize = (12,6))

sns.countplot(
    data = df,
    x = "Payment Method",
    hue = "Churn Label"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)    # Rotate x-axis labels for better readability
plt.legend(title = "Churn")

plt.tight_layout()
plt.show()

# Calculate the Churn Rate by Payment Method
payment_churn_rate = (
    df.groupby("Payment Method")["Churn Value"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nChurn Rate by Payment Method: ")
print(payment_churn_rate)

# Understand the values in senior citizen column
print("\nSenior Citizen Values: ")
print(df["Senior Citizen"].value_counts())

# Calculate the Churn Rate by Senior Citizen status
senior_churn_rate = (
    df.groupby("Senior Citizen")["Churn Value"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nChurn Rate by Senior Citizen Status: ")
print(senior_churn_rate)

# Visualize the relationship between Senior Citizen status and Churn
plt.figure(figsize = (8,6))

sns.barplot(
    x = senior_churn_rate.index,
    y = senior_churn_rate.values
)

plt.title("Churn Rate by Senior Citizen Status")
plt.xlabel("Senior Citizen Status (0 = No, 1 = Yes)")
plt.ylabel("Churn Rate (%)")

for i, value in enumerate(senior_churn_rate.values):
    plt.text(
        i,
        value + 1,  # Position the text slightly above the bar
        f'{value:.2f}%',  # Format the value as a percentage with two decimal places
        ha='center'
)

plt.tight_layout()
plt.show()

# Visualize the relationship between Partners and Dependents with Churn
def plot_churn_rate(column,title):
    churn_rate = (
        df.groupby(column)["Churn Value"]
        .mean()
        .sort_values(ascending=False)
        *100
    )

    print(f"\nChurn Rate by {column}: ")
    print(churn_rate)

    plt.figure(figsize = (8,6))

    sns.barplot(
        x = churn_rate.index,
        y = churn_rate.values
    )

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Churn Rate (%)")

    for i, value in enumerate(churn_rate.values):
        plt.text(
            i,
            value + 1,  # Position the text slightly above the bar
            f'{value:.2f}%',  # Format the value as a percentage with two decimal places
            ha='center'
        )

    plt.tight_layout()
    plt.show()

plot_churn_rate("Partner","Churn Rate by Partner Status")

plot_churn_rate("Dependents","Churn Rate by Dependents Status")

# Visualize the relationship between Online Security and Tech Support with Churn

plot_churn_rate("Online Security","Churn Rate by Online Security Status")

plot_churn_rate("Tech Support","Churn Rate by Tech Support Status")

# Visualize the relationship between paperless billing and Churn

plot_churn_rate("Paperless Billing","Churn Rate by Paperless Billing Status")

# Check the data type of Total Charges column
print("\nData Type of Total Charges Column: ")
print(df["Total Charges"].dtype)

# Convert Total Charges to numeric, coercing errors to NaN
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], 
    errors='coerce'
)

# Check for missing values in Total Charges after conversion
print("\nMissing Values in Total Charges after Conversion: ")
print(df["Total Charges"].isnull().sum())

# Visualize the relationship between Total Charges and Churn
plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x = "Churn Label",
    y = "Total Charges"
)

plt.title("Customer Churn by Total Charges")
plt.xlabel("Churn Label")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.show()

# Identify the numerical columns for correlation analysis

print("\nNumerical Columns: ")
print(df.select_dtypes(include = "number").columns)

# Correlation matrix for numerical features

numeric_df = df.select_dtypes(include = "number")
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True
)

plt.title("Correlation Matrix of Numerical Features")

plt.tight_layout()
plt.show()