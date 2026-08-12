import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


from data_loader import load_data
from feature_engineering import add_features


# Load the data set
df = load_data()

# Check the data set shape
print("\nDataset Shape: ")
print(df.shape)

# Check Data Types
print("\nData Types: ")
print(df.dtypes)

# Check for missing values
print("\nMissing Values: ")
print(df.isnull().sum())

# Check Type of Total Charges
print("\nType of Total Charges: ")
print(type(df["Total Charges"].dtype))

# Concert Total Charges to Numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors = "coerce"
)

# Check again for the data type after conversion
print("\nData Type of Total Charges after Conversion: ")
print(df["Total Charges"].dtype)

# Check Missing values again
print("\nMissing Values after Conversion: ")
print(df.isnull().sum())

# Handle the missing Total Charges
df = df.dropna(subset = ["Total Charges"])

# Check for duplicates
print("\nDuplicate Rows: ")
print(df.duplicated().sum())

# Remove Duplicate rows.
df = df.drop_duplicates()

print("\nDataset Shape after Removing Duplicates: ")
print(df.shape)

# Examine dataset after removing missing values
print("\nDataset Shape after Removing Missing Values in Total Charges: ")
print(df.shape)

# Remove columns that should not be used for modeling.
columns_to_drop = [
    "CustomerID",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "CLTV"
]

df = df.drop(columns = columns_to_drop, errors = "ignore")

# Add engineered features.
df = add_features(df)

# Check the newly created features
print("\nEngineered Features: ")
print(df[
    [
        "Additional Services Count",
        "Tenure Group",
        "New Customer + Month-to-Month"
    ]
].head())

# Check for the new columns 
print("\nDataset Columns after Feature Engineering: ")
print(df.columns.tolist())

# Prepare target variable
df["Churn Label"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0 
})

# Check for the counts of the target variable
print("\nChurn Distribution: ")
print(df["Churn Label"].value_counts())

# Check Correlation between numerical features and Churn
print("\nNumerical Feature Correlation with Churn: ")
print(
    df.select_dtypes(include = "number")
    .corr()["Churn Label"]
    .sort_values(ascending = False)
)

# Seperate X and Y
# X - Everything used to make the prediction (Customer Information)
# Y - The target variable we are trying to predict (Churn Label)

x = df.drop(columns = ["Churn Label"])
y = df["Churn Label"]

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

# Identify Numerical features
numerical_features = x_train.select_dtypes(
    include = ["int64", "float64"]
).columns.tolist()

# Print the numerical features
print("\nNumerical Features: ")
print(numerical_features)

# Identify Categorical features.
categorical_features = x_train.select_dtypes(
    include = ["object","str"]
).columns.tolist()

# print Categorica features
print("\nCategorical Features: ")
print(categorical_features)

# Create the Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers = [
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown = "ignore"),
            categorical_features
        )
    ]
)

# Apply preprocessing
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

# Create directory for saved preprocessing pipeline.
os.makedirs("C:/itsChutiii/Code & Cry (Uni)/My projects/Data Science/Customer-Churn-Prediction-System/models", exist_ok = True)

# Save the fitted preprocessing pipeline
joblib.dump(
    preprocessor,
    "C:/itsChutiii/Code & Cry (Uni)/My projects/Data Science/Customer-Churn-Prediction-System/models/preprocessor.pkl"
)

print("\nPreprocessing pipeline saved successfully.")

# Check Results
print("\nOriginal Training Shape:")
print(x_train.shape)

print("\nProcessed Training Shape:")
print(x_train_processed.shape)

print("\nOriginal Testing Shape:")
print(x_test.shape)

print("\nProcessed Testing Shape:")
print(x_test_processed.shape)

# Create processed data directory.

processed_data_path = "C:/itsChutiii/Code & Cry (Uni)/My projects/Data Science/Customer-Churn-Prediction-System/data/processed"

os.makedirs(processed_data_path, exist_ok=True)

# Convert Processed matrices to Dataframes

x_train_processed_df = pd.DataFrame(
    x_train_processed,
    columns=preprocessor.get_feature_names_out()
)

x_test_processed_df = pd.DataFrame(
    x_test_processed,
    columns=preprocessor.get_feature_names_out()
)

# Save processed training and testing features

x_train_processed_df.to_csv(
    f"{processed_data_path}/x_train_processed.csv",
    index = False
)

x_test_processed_df.to_csv(
    f"{processed_data_path}/x_test_processed.csv",
    index = False
)

# save target varaibles

y_train.to_csv(
    f"{processed_data_path}/y_train.csv",
    index = False
)

y_test.to_csv(
    f"{processed_data_path}/y_test.csv",
    index = False
)

print("\nProcessed dataset Saved Successfully.")

print("\nSaved Files: ")
print("X_train_processed.csv")
print("X_test_processed.csv")
print("y_train.csv")
print("y_test.csv")