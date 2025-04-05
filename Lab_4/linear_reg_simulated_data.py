import numpy as np
import pandas as pd
from gradient_descent_scratch import LinearRegression
from linear_reg_evaluation_metrics import RegressionMetrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data):
    """
    Load dataset from a CSV file and display the first few rows.

    Parameters:
        data (str): File path of the dataset.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(data)
    print("\nDataset Loaded Successfully!")
    print("-" * 50)
    print(df.head())  # Display first 5 rows
    print("-" * 50)

    return df

def eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.

    Parameters:
        data (str): File path of the dataset.
    """
    print("\nPerforming Exploratory Data Analysis...")
    df = load_data(data)

    print("\nDataset Information:")
    print("-" * 50)
    print(df.info())

    print("\nSummary Statistics:")
    print("-" * 50)
    print(df.describe())

    print("\nChecking Missing Values:")
    print("-" * 50)
    print(df.isnull().sum())

    print("\nChecking Duplicates:")
    print("-" * 50)
    print(f"Total Duplicated Rows: {df.duplicated().sum()}")

def split_and_standardize(data):
    """
    Split the dataset into training and testing sets and apply feature scaling.

    Parameters:
        data (str): File path of the dataset.

    Returns:
        tuple: Scaled training & testing data (X_train, X_test, y_train, y_test)
    """
    print("\n✂Splitting Data into Train & Test Sets...")
    df = load_data(data)

    X = df.drop(columns=['disease_score'], axis=1)  # Independent variables
    y = df['disease_score']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Data Splitting Completed! Training Size:", X_train.shape, "Testing Size:", X_test.shape)

    # Standardizing the features (mean = 0, std = 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature Scaling Applied (Standardization Done)")

    return X_train_scaled, X_test_scaled, y_train, y_test

def linear_regression(data):
    """
    Train a Linear Regression model using Gradient Descent and evaluate performance.

    Parameters:
        data (str): File path of the dataset.
    """
    print("\nTraining Linear Regression Model...")
    X_train, X_test, y_train, y_test = split_and_standardize(data)

    # Convert target variable to NumPy array
    y_train = np.array(y_train)

    # Initialize Linear Regression model
    linear_model = LinearRegression(learning_rate=0.01)

    # Train the model
    linear_model.fit(X_train, y_train)
    print("Model Training Completed!")

    # Make predictions
    y_pred = linear_model.predict(X_test)
    
    # Evaluate model performance
    r2_score = metrics.r_squared(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"R² Score: {r2_score:.4f}")  # R-squared score (higher is better)
    print("-" * 50)

    return y_test,y_pred

def plot_predictions(y_true, y_pred):
    """
    Plot predicted vs actual target values.

    Parameters:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='teal', edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Disease Score')
    plt.ylabel('Predicted Disease Score')
    plt.title('Actual vs Predicted Disease Score (Gradient Descent)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the workflow.
    """
    data = r"C:/Users/Naga Nandi Reddy/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    print("\nLoading Dataset from:", data)
    eda(data)  # Perform EDA
    y_test,y_pred=linear_regression(data)  # Train & Evaluate Model
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
