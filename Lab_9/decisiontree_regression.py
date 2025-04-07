import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from sklearn.model_selection import train_test_split, \
    KFold  # Import functions to split data and perform cross-validation
from sklearn.preprocessing import StandardScaler  # For scaling the features
from sklearn.tree import DecisionTreeRegressor  # For building the decision tree regressor
from sklearn.metrics import r2_score  # To evaluate model performance using accuracy score


# Function to load the dataset, preprocess, and split into train and test sets
def load_data(data):
    # Load the data from CSV file
    df = pd.read_csv(data)

    # Print the first 5 rows of the dataset to inspect the data
    print(df.head())

    # Split the dataset into training (70%) and testing (30%) sets
    x_train, x_test = train_test_split(df, test_size=0.3, random_state=42)

    return x_train, x_test


# Function to perform hyperparameter tuning (select the best max_depth for the decision tree)
def hyper_tune(x_train, x_val, y_train, y_val):
    max_depths = [1, 2, 3]  # List of potential maximum depths for the tree
    max_score = 0  # To store the best accuracy score achieved
    opt_depth = None  # To store the optimal depth for the decision tree

    # Iterate through each possible maximum depth
    for depth in max_depths:
        clf = DecisionTreeRegressor(max_depth=depth)  # Create a decision tree with the specified depth
        clf.fit(x_train, y_train)  # Train the classifier on the training data
        y_pred = clf.predict(x_val)  # Make predictions on the validation set
        fold_score = r2_score(y_val, y_pred)

        # If the current fold score is better than the previous best, update the best score and optimal depth
        if fold_score > max_score:
            max_score = fold_score
            opt_depth = depth

    # Return the optimal depth for the decision tree
    return opt_depth


# Function to build and evaluate a decision tree classifier using cross-validation
def decision_tree(data):
    # Load and preprocess the data
    x_train, x_test = load_data(data)

    # Perform K-fold cross-validation (5 splits)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Separate features (x) and target (y) in the training data
    x = x_train.drop(columns=['disease_score'], axis=1)  # Features (drop the target 'diagnosis')
    y = x_train['disease_score']  # Target variable (diagnosis)

    # Scale the features using StandardScaler to ensure they are on the same scale
    sc = StandardScaler()
    x_tr = sc.fit_transform(x)  # Apply scaling to the features

    scores = []  # List to store accuracy scores for each fold
    best_clf = None  # Variable to store the best classifier found during cross-validation

    # Perform K-fold cross-validation
    for i, (train_index, val_index) in enumerate(kf.split(x_tr, y)):
        print("Fold: ", i)  # Print the current fold number

        # Split the data into training and validation sets for this fold
        x_train_fold, x_val = x_tr[train_index], x_tr[val_index]
        y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

        # Find the optimal maximum depth for the decision tree using hyperparameter tuning
        opt_depth = hyper_tune(x_train_fold, x_val, y_train_fold, y_val)
        print("Optimal Depth: ", opt_depth)

        # Train a decision tree classifier with the optimal depth
        clf = DecisionTreeRegressor(max_depth=opt_depth)
        clf.fit(x_train_fold, y_train_fold)  # Train on the current fold's training data
        y_pred = clf.predict(x_val)  # Predict on the validation data

        # Calculate r2 score for the fold and append it to the list of scores
        score = r2_score(y_val, y_pred)
        scores.append(score)

        # If this is the last fold (i == 4), save the best classifier
        if i == 4:
            best_clf = clf

    # Calculate and print the mean and standard deviation of accuracy scores across all folds
    mean_score = np.mean(scores)
    print("Mean r2_score:", mean_score)
    std = np.std(scores)
    print("Standard Deviation:", std)

    # Evaluate the final model on the test set
    x_test_scaled = sc.transform(x_test.drop(columns=['disease_score'], axis=1))  # Scale the test data
    y_test = x_test['disease_score']  # Extract the target values from the test data
    y_test_pred = best_clf.predict(x_test_scaled)  # Predict on the test set

    # Calculate the r2 score on the test data
    test_score = r2_score(y_test, y_test_pred)
    print("Test r2_score:", test_score)


# Main function to load the dataset and start the decision tree process
def main():
    # Specify the path to the dataset (CSV file)
    df = r"C:/Users/Naga Nandi Reddy/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"

    # Call the decision_tree function with the path to the dataset
    decision_tree(df)

if __name__ == '__main__':
    main()
