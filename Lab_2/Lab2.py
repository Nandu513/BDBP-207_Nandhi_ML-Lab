# Import necessary libraries and modules
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and test sets
from sklearn.linear_model import LinearRegression  # For performing linear regression
from sklearn.metrics import r2_score  # For evaluating the model's performance using R^2 score
from sklearn.datasets import fetch_california_housing  # To fetch the California housing dataset
import matplotlib.pyplot as plt  # For data visualization (although not used in the current code)
from sklearn.preprocessing import StandardScaler  # For scaling (standardizing) the features

# Function to load the California housing dataset
def load_data():
    # Fetches the dataset and splits it into features (X) and target (y)
    [x, y] = fetch_california_housing(return_X_y=True)
    return (x, y)

# Main function to train and evaluate the linear regression model
def main():
    # Load the data
    [x, y] = load_data()

    # Split the data into training and test sets (70% training, 30% testing)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=999)

    # Initialize the StandardScaler for feature scaling
    scalar = StandardScaler()

    # Fit the scaler on the training data and transform both the training and test data
    scaler = scalar.fit(x_train)  # Fit the scaler to the training data
    x_train_scaler = scaler.transform(x_train)  # Apply the scaling to the training data
    x_test_scaler = scalar.transform(x_test)  # Apply the scaling to the test data

    # Initialize the linear regression model
    reg_model = LinearRegression()

    # Train the model using the scaled training data
    reg_model.fit(x_train_scaler, y_train)

    # Predict the target values for the test data
    y_pred = reg_model.predict(x_test_scaler)

    # Calculate the R^2 score to evaluate the model's performance
    r2 = r2_score(y_test, y_pred)

    # Print the R^2 score
    print(f"R^2 score: {r2}")
    print("Coefficients (theta values):", reg_model.coef_)

# Entry point for the script
if __name__ == '__main__':
    main()
