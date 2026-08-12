import pandas as pd

# Load the data from a CSV file
def load_data():
    file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(file_path)
    return df

# Check the data set is loaded correctly
if __name__ == "__main__":
    df = load_data()
    print("Dataset loaded successfully!")
    print(df.head())

    
    