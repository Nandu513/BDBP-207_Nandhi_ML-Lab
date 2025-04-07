import numpy as np

def entropy(y):
    """
    Computes the entropy of the given label values.

    Parameters:
        y (ndarray): Input label values.

    Returns:
        entropy (float): Entropy of the given label values.
    """
    entropy = 0

    # Find the unique label values in y and loop over each value
    labels = np.unique(y)
    for label in labels:
        # Find the examples in y that have the current label
        label_examples = y[y == label]
        # Calculate the ratio of the current label in y
        pl = len(label_examples) / len(y)
        # Calculate the entropy using the current label and ratio
        entropy += -pl * np.log2(pl)

    # Return the final entropy value
    return entropy


def information_gain(parent, left, right):
    """
    Computes the information gain from splitting the parent dataset into two datasets.

    Parameters:
        parent (ndarray): Input parent dataset.
        left (ndarray): Subset of the parent dataset after split on a feature.
        right (ndarray): Subset of the parent dataset after split on a feature.

    Returns:
        information_gain (float): Information gain of the split.
    """
    # set initial information gain to 0
    information_gain = 0
    # compute entropy for parent
    parent_entropy = entropy(parent)
    # calculate weight for left and right nodes
    weight_left = len(left) / len(parent)
    weight_right = len(right) / len(parent)
    # compute entropy for left and right nodes
    entropy_left, entropy_right = entropy(left), entropy(right)
    # calculate weighted entropy
    weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
    # calculate information gain
    information_gain = parent_entropy - weighted_entropy
    return information_gain


def main():
    # Dummy class labels
    parent = ['yes', 'yes', 'no', 'no', 'yes']
    left = ['yes', 'yes']
    right = ['no', 'no', 'yes']

    print("Entropy of parent:", entropy(parent))
    print("Information Gain:", information_gain(parent, left, right))

if __name__=="__main__":
    main()
