import pandas as pd  # Import pandas for data manipulation

# Function to split the DataFrame based on a threshold for a given column
def split_threshold(df, threshold, column):
    # Split the DataFrame into two parts based on the threshold for the specified column:
    # - df_left: Rows where the value in the column is less than or equal to the threshold
    # - df_right: Rows where the value in the column is greater than the threshold
    df_left = df[df[column] <= threshold]
    df_right = df[df[column] > threshold]

    # Print the shape (number of rows and columns) of the left and right split DataFrames
    print(f"Rows in left split (<= {threshold}): {df_left.shape}")
    print(f"Rows in right split (> {threshold}): {df_right.shape}")

    # Return the two split DataFrames for further use
    return df_left, df_right

# Main function to interact with the user, load the data, and call the split_threshold function
def main():
    # Specify the path to the dataset (CSV file)
    df = "/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    
    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(df)
    
    # Print the first 5 rows of the dataset to inspect the data
    print(data.head())
    
    # Prompt the user to input the threshold value for splitting the data
    threshold = int(input("Enter the threshold: "))
    
    # Prompt the user to input the column name on which to base the split
    column = input("Enter the column: ")
    
    # Call the split_threshold function to split the data based on the user's inputs
    split_threshold(data, threshold, column)

# Entry point of the script
if __name__ == "__main__":
    main()
