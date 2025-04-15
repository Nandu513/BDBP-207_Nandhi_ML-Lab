import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 1. Generate synthetic regression data
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train multiple trees
n_trees = 5
trees = []

for i in range(n_trees):
    tree = DecisionTreeRegressor(max_depth=3, random_state=i)

    # Bootstrap sampling (optional, for more randomness)
    sample_indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
    x_sample = x_train[sample_indices]
    y_sample = y_train[sample_indices]

    tree.fit(x_sample, y_sample)
    trees.append(tree)


# 3. Aggregate predictions by averaging
def aggregate_predictions(trees, X):
    all_preds = np.array([tree.predict(X) for tree in trees])
    return np.mean(all_preds, axis=0)


# Get predictions
y_pred = aggregate_predictions(trees, x_test)

# 4. Evaluation
print("Aggregated Model Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# 5. Visualization
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Aggregated Trees)")
plt.tight_layout()
plt.show()
