import numpy as np
import pandas as pd


# Function to load and inspect the data
def load_data(data):
    df = pd.read_csv(data)
    print(df.head())
    print(df.shape)
    return df


# Node class to represent each node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, gain=None, left=None, right=None, node_value=None):
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right
        self.node_value = node_value


# Function to calculate entropy
def find_entropy(y):
    entropy = 0
    labels = np.unique(y)
    for label in labels:
        label_set = y[y == label]
        pi = len(label_set) / len(y)
        entropy += -pi * np.log2(pi)
    return entropy


# Function to split the data based on feature and threshold
def split_data(data, feature, threshold):
    left_data = data[data[feature] <= threshold]
    right_data = data[data[feature] > threshold]
    return left_data, right_data


# Function to calculate the information gain
def find_gain(parent, left, right):
    parent_entropy = find_entropy(parent)
    weight_left = len(left) / len(parent)
    weight_right = len(right) / len(parent)
    entropy_left = find_entropy(left)
    entropy_right = find_entropy(right)
    expected_entropy = weight_left * entropy_left + weight_right * entropy_right
    information_gain = parent_entropy - expected_entropy
    return information_gain


# DecisionTree class to build and train the tree
class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    # Function to find the best split for the data
    def best_split(self, data, n_samples, n_features):
        best_split = {'gain': -1, 'feature': None, 'threshold': None, 'left_dataset': None, 'right_dataset': None}

        for feature in range(n_features):
            feature_values = np.unique(data[:, feature])
            for threshold in feature_values:
                left_data, right_data = split_data(data, feature, threshold)

                # Only consider splits where both left and right datasets are non-empty
                if len(left_data) > 0 and len(right_data) > 0:
                    gain = find_gain(data[:, -1], left_data[:, -1], right_data[:, -1])

                    # Update the best split if the current one has a higher information gain
                    if gain > best_split['gain']:
                        best_split['gain'] = gain
                        best_split['feature'] = feature
                        best_split['threshold'] = threshold
                        best_split['left_dataset'] = left_data
                        best_split['right_dataset'] = right_data

        return best_split

    # Function to calculate the most common label for leaf nodes
    @staticmethod
    def calculate_leaf_value(y):
        y = list(y)
        most_occurring_value = max(y, key=y.count)
        return most_occurring_value

    # Function to build the decision tree recursively
    def build_tree(self, data, current_depth=0):
        n_samples, n_features = data.shape

        # Base condition: stop if the tree is too deep or not enough samples
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best_splitting = self.best_split(data, n_samples, n_features)

            if best_splitting["gain"] > 0:  # Proceed only if there is a valid split
                left_node = self.build_tree(best_splitting["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_splitting["right_dataset"], current_depth + 1)
                return Node(best_splitting["feature"], best_splitting["threshold"],
                            best_splitting["gain"], left_node, right_node)

        # If no valid split, return a leaf node
        leaf_value = self.calculate_leaf_value(data[:, -1])
        return Node(node_value=leaf_value)

    # Function to train the decision tree
    def fit(self, data):
        self.root = self.build_tree(data)

    # Function to make predictions on new data
    def predict(self, data):
        predictions = [self._predict_single(x) for x in data]
        return predictions

    def _predict_single(self, x, node=None):
        if node is None:
            node = self.root

        # If we have reached a leaf, return the node's value
        if node.node_value is not None:
            return node.node_value

        # Otherwise, go left or right based on the threshold
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


# Main function to run the code
def main():
    df = load_data("/home/ibab/Downloads/data.csv")

    # Preprocessing the dataset (Assuming target is the last column)
    data = df.values  # Convert to numpy array
    tree = DecisionTree(min_samples=10, max_depth=5)
    tree.fit(data)

    # To predict with the trained model:
    predictions = tree.predict(data[:,:-1])  # Use all columns except the last (target)

    print(predictions)  # Print predictions


if __name__ == "__main__":
    main()
