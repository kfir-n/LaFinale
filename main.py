# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

# Read the table into a pandas DataFrame
# df is a global variable
Database_File = pd.read_csv('Database/Database_Original.csv')
df_dir = 'Database/Database_Original.csv'
num_children = 15


def main():
    # printing the original table
    print_csv_table(df_dir)
    print(f"\n\n\n****************************************\n****************************************")
    print(f"\t\tafter changes")
    print(f"\t\tafter changes")
    print(f"****************************************\n****************************************")
    print(f"\nafter changes")

    Database_File['No'] = range(1,len(generate_random_data(num_children)+1))

    excel_file_path = 'Database/Database_new.csv'
    Database_File.to_csv(excel_file_path, index=False)

    # print_csv_table(df)
    print_csv_table(excel_file_path)


def generate_random_data(num_rows):
    """
    Generates a DataFrame with random data according to specified conditions.
    - 'S' and 'R4s' will be randomly set to 1 or 2.
    - All other values will be random numbers between 0 and 100.
    - 'R4s' will change to 2 only if the average in the row (excluding 'S' and 'R4s') is greater than 55.

    Parameters:
    - num_rows: The number of rows to generate.

    Returns:
    - A pandas DataFrame with the generated data.
    """
    columns = [
        "No",
        "S",
        "Arith",
        "Reading",
        "Dictation",
        "Exercise",
        "Havanat",
        "Analogy",
        "Arithmetic Operations",
        "Baloon Counting",
        "Chose the form",
        "Click the …",
        "Count and numbers",
        "Faces",
        "Incomplete shadow",
        "Magic Circle",
        "More or less",
        "Remember location",
        "Set order",
        "Triangles",
        "R4s"
    ]

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=columns)

    for i in range(num_rows):
        # Generate random numbers for each column, excluding 'S' and 'R4s'
        row_data = np.random.randint(0, 101, size=len(columns) - 2)
        # Calculate the average of the generated numbers
        average = np.mean(row_data)
        # Determine 'S' and 'R4s' values
        s_value = np.random.choice([1, 2])
        r4s_value = 2 if average > 55 else 1
        # Combine all values into a single row
        row = np.append(np.array([s_value]), np.append(row_data, r4s_value))
        # Add the row to the DataFrame
        df.loc[i] = row

    return df


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


if __name__ == "__main__":
    main()
