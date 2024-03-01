# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Read the table into a pandas DataFrame
# df is a global variable
Database_File = pd.read_csv('Database/Database_Original.csv')
df_dir = 'Database/Database_Original.csv'
num_children = 15

# Define the file path, target column, and any categorical columns needing encoding
categorical_columns = ['S', 'r4s']  # Replace or modify according to your dataset's needs


# Call the function


def main():
    # printing the original table
    # Assignment 1
    print_csv_table(df_dir)
    print(f"\n\n\n****************************************\n****************************************")
    print(f"\t\tafter changes")
    print(f"\t\tafter changes")
    print(f"****************************************\n****************************************")
    print(f"\nafter changes")
    fill_missing_values(df_dir, 'Database/new_file.csv')

    # labeled encoder
    df = pd.read_csv('Database/new_file.csv')

    # Assignment 2
    # Specify the columns you want to encode
    categorical_columns = ['S', 'r4s']  # Add any other categorical columns as needed
    target_column = 'S'  # actual target column name

    # Encode the specified columns
    df_encoded = encode_categorical_columns(df, categorical_columns)

    # Show the first few rows of the encoded DataFrame
    print(df_encoded.head())

    # Optionally, save the encoded DataFrame to a new CSV file
    df_encoded.to_csv('encoded_data.csv', index=False)

    # Assignment 3
    # Splitting into dataSet and testSet
    X_train, X_test, y_train, y_test = preprocess_and_split_data('Database/new_file.csv', target_column,
                                                                 categorical_columns)

    # Print shapes of the datasets
    print("Shapes of the datasets:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Print the first few rows of each dataset to verify
    print("\nFirst few rows of X_train:")
    print(X_train.head())
    print("\nFirst few rows of X_test:")
    print(X_test.head())
    print("\nFirst few rows of y_train:")
    print(y_train.head())
    print("\nFirst few rows of y_test:")
    print(y_test.head())


def preprocess_and_split_data(file_path, target_column, categorical_columns=None, test_size=0.2, random_state=42):
    """
    Preprocess the dataset by encoding categorical columns and then split it into training and test sets.

    Parameters:
    - file_path: str - The path to the CSV file.
    - target_column: str - The name of the target variable column.
    - categorical_columns: list - A list of column names to be label encoded. If None, no encoding is applied.
    - test_size: float - The proportion of the dataset to include in the test split.
    - random_state: int - A seed used by the random number generator for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: The split datasets.
    """

    # Load the dataset
    df = pd.read_csv(file_path)

    # Encode categorical columns if specified
    if categorical_columns is not None:
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            df[column] = df[column].astype(str)  # Convert to string to handle mixed types
            df[column] = label_encoder.fit_transform(df[column])

    # Separate features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def generate_random_date(year):
    """Generate a random date within a given year."""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + pd.Timedelta(days=random_number_of_days)
    return random_date.strftime('%d-%m-%Y')


def fill_missing_values(file_path, output_file_path):
    df = pd.read_csv(file_path)

    for column in df.columns:
        if column == 'B.D':
            df[column] = df[column].apply(lambda x: generate_random_date(1997) if pd.isnull(x) else x)
        elif column in ['S', 'r4s']:
            df[column] = df[column].fillna(random.choice([1, 2]))
        else:
            # For numerical columns
            df[column] = df[column].apply(lambda x: round(random.uniform(0, 100), 2) if pd.isnull(x) else x)

    # Adjust R4s based on row average > 55
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('S')  # Assuming 'S' is not part of the average calculation
    numeric_cols.remove('r4s')  # 'R4s' should not be part of its own calculation
    for index, row in df.iterrows():
        row_average = row[numeric_cols].mean()
        if row_average > 55:
            df.at[index, 'r4s'] = 2
        else:
            df.at[index, 'r4s'] = 1

    # Save to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"File has been saved to {output_file_path} with missing values filled.")


# Example usage

def print_csv_table(file_path):
    """
    Read a CSV file and print its contents in table format.

    Args:
        file_path (str): File path of the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(df.to_string(index=False))  # Print DataFrame as table without index
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def encode_categorical_columns(df, columns):
    """
    Apply label encoding to specified columns of a pandas DataFrame.

    Parameters:
    - df: pandas.DataFrame - The DataFrame containing the data.
    - columns: list - A list of column names to be label encoded.

    Returns:
    - df_encoded: pandas.DataFrame - The DataFrame with specified columns label encoded.
    """

    # Make a copy of the DataFrame to avoid modifying the original data
    df_encoded = df.copy()

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Apply Label Encoding to each specified column
    for column in columns:
        # Convert to string to handle mixed types and missing values
        df_encoded[column] = df_encoded[column].astype(str)

        # Apply LabelEncoder and replace the column in the DataFrame
        df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    return df_encoded


# Example usage


if __name__ == "__main__":
    main()
