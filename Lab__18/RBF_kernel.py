import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

# Step 1: Create the dataset 
data = {
    'x1': [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
    'x2': [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8],
    'Label': ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Red',
              'Red', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
}

df = pd.DataFrame(data)

# Convert labels to numeric
df['y'] = df['Label'].map({'Blue': 0, 'Red': 1})

X = df[['x1', 'x2']].values
y = df['y'].values

# Step 2: Plot function for decision boundary 
def plot_svm_boundary(X, y, kernel_type, ax):
    clf = SVC(kernel=kernel_type, C=1, gamma='scale')  # 'scale' is default and usually good
    clf.fit(X, y)

    # Plot decision boundary and margins
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        shading='auto'
    )

    # Plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=120, facecolors='none', edgecolors='k', label='Support Vectors')

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=80, edgecolors='k')
    ax.set_title(f"SVM with {kernel_type} kernel")
    ax.legend(*scatter.legend_elements(), title="Class")

# Step 3: Plot both kernels side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_svm_boundary(X, y, 'poly', axs[0])  # Polynomial kernel
plot_svm_boundary(X, y, 'rbf', axs[1])   # RBF kernel

plt.tight_layout()
plt.show()
