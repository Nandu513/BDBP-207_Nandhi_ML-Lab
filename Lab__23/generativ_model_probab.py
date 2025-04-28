import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP import load_data


def load_and_preprocess_data():
    # Load the Iris dataset from sklearn
    iris = load_iris()

    # Convert the data into a Pandas DataFrame with the appropriate feature names
    df1 = pd.DataFrame(iris.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    df2 = pd.DataFrame(iris.target, columns=['Species'])

    # Select only the first two features and the target
    X = df1[['SepalLengthCm', 'SepalWidthCm']]
    y = df2['Species']

    # Add random noise to the features
    noise = np.random.normal(0, 0.1, X.shape)  # Gaussian noise with mean=0 and std=0.1
    X_noisy = X + noise

    return X_noisy, y

def discretize_features(X, bins=5):
    # Discretize the features into `bins` equal-width bins
    X_discretized = X.apply(lambda x: pd.cut(x, bins=bins, labels=False))
    return X_discretized


def build_decision_tree(X_train, y_train, X_test, y_test):
    # Train a decision tree model
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def joint_probability_distribution(X_train, y_train):
    # Discretize the features
    X_discretized = discretize_features(X_train)

    # Get the unique values for each feature
    feature_combinations = pd.DataFrame(list(X_discretized.itertuples(index=False)),
                                        columns=['SepalLengthCm', 'SepalWidthCm'])
    feature_combinations['Species'] = y_train

    # Compute the joint probability distribution
    joint_prob = feature_combinations.groupby(['SepalLengthCm', 'SepalWidthCm', 'Species']).size().reset_index(
        name='count')
    total_count = joint_prob['count'].sum()
    joint_prob['probability'] = joint_prob['count'] / total_count

    return joint_prob


def predict_with_joint_probability(X_test, joint_prob):
    # Discretize the test set features
    X_discretized = discretize_features(X_test)

    # For each row in the test set, calculate the class with the highest joint probability
    predictions = []
    for _, row in X_discretized.iterrows():
        # Get the joint probability for the given feature combination
        prob_row = joint_prob[(joint_prob['SepalLengthCm'] == row['SepalLengthCm']) &
                              (joint_prob['SepalWidthCm'] == row['SepalWidthCm'])]

        if prob_row.empty:
            predictions.append(None)  # In case the combination doesn't exist in training data
        else:
            # Choose the class with the highest probability
            predictions.append(prob_row.sort_values(by='probability', ascending=False).iloc[0]['Species'])

    return predictions


def evaluate_models(X_train, X_test, y_train, y_test):
    # Build and evaluate the decision tree model
    dt_accuracy = build_decision_tree(X_train, y_train, X_test, y_test)
    print(f"Decision Tree Model Accuracy: {dt_accuracy:.4f}")

    # Calculate the joint probability distribution
    joint_prob = joint_probability_distribution(X_train, y_train)

    # Predict using joint probability distribution
    y_pred_joint_prob = predict_with_joint_probability(X_test, joint_prob)

    # Evaluate accuracy of the joint probability distribution method
    joint_prob_accuracy = accuracy_score(y_test, y_pred_joint_prob)
    print(f"Joint Probability Distribution Model Accuracy: {joint_prob_accuracy:.4f}")


def main():
    # Path to the Iris dataset
    file_path = 'Iris.csv'  # Modify to the correct path of your file

    # Load and preprocess the data
    X, y = load_and_preprocess_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Evaluate the models
    evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
