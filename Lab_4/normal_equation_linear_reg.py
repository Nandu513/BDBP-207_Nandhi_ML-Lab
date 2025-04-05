import numpy as np
import pandas as pd
from numpy import linalg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import process_time


def calculate_theta(X, y):
    """
    Calculates the theta vector using the Normal Equation.
    """
    X_T = X.transpose()
    try:
        theta = np.linalg.inv(X_T @ X) @ X_T @ y
    except np.linalg.LinAlgError:
        print("Matrix not invertible. Using pseudo-inverse instead.")
        theta = np.linalg.pinv(X_T @ X) @ X_T @ y
    return theta.reshape(-1)


def hypothesis(X, theta):
    """
    Computes predictions (hypothesis) given input X and theta.
    """
    return X @ theta


def compute_cost(X, y, theta):
    """
    Computes the mean squared error cost function.
    """
    m = len(y)
    errors = hypothesis(X, theta) - y.reshape(-1)
    return (1 / (2 * m)) * (errors.T @ errors)


def add_intercept_column(X):
    """
    Adds a column of 1s to X for the intercept term (X_0 = 1).
    """
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))


def print_separator(title=None):
    print("\n" + "-" * 60)
    if title:
        print(f"{title}")
        print("-" * 60)


def load_and_split_data(path):
    """
    Loads data and returns processed training and validation sets.
    """
    df = pd.read_csv(path)

    X = df.drop(columns=['disease_score']).values
    y = df['disease_score'].values

    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)



def main():
    # Load data
    data_path = r"C:/Users/Naga Nandi Reddy/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    X_train, X_valid, y_train, y_valid = load_and_split_data(data_path)

    # Add intercept term (X_0 = 1)
    X_train_b = add_intercept_column(X_train)
    X_valid_b = add_intercept_column(X_valid)

    # Start training timer
    start_time = process_time()

    # Train using Normal Equation
    theta = calculate_theta(X_train_b, y_train)

    # End training timer
    elapsed_time = (process_time() - start_time) * 1000

    # Print results
    print_separator("Training Summary")
    print(f"Theta (parameters):\n{theta}")
    print(f"Training Time: {elapsed_time:.4f} ms")

    # Cost on training and validation sets
    cost_train = compute_cost(X_train_b, y_train, theta)
    cost_valid = compute_cost(X_valid_b, y_valid, theta)

    print_separator("Model Performance")
    print(f"Training Cost: {cost_train:.4f}")
    print(f"Validation Cost: {cost_valid:.4f}")

    # First 5 predictions vs actuals
    y_pred = hypothesis(X_valid_b, theta)
    results = pd.DataFrame({
        "Actual disease_score": y_valid[:5],
        "Predicted disease_score": y_pred[:5]
    })

    print_separator("Predictions vs Actuals (first 5)")
    print(results.to_string(index=False))

    # Optional: plot actual vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_valid, y_pred, alpha=0.7, color="green")
    plt.plot([0, 100], [0, 100], '--', color='gray')
    plt.xlabel("Actual disease_score")
    plt.ylabel("Predicted disease_score")
    plt.title("Actual vs Predicted (Validation Set)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
