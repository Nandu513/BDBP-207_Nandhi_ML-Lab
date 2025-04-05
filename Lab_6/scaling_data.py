import numpy as np

def normalize(data):
    """
    Normalize data to range [0, 1].
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def standardize(data):
    """
    Standardize data to zero mean and unit variance.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data


# Sample test data
X = np.array([[1, 200],
              [2, 300],
              [3, 400],
              [4, 500]])

# Normalize
X_norm = normalize(X)
print("Normalized:\n", X_norm)

# Standardize
X_std = standardize(X)
print("Standardized:\n", X_std)
