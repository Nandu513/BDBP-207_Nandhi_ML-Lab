import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 40
n_features = 50

# Generate three classes with mean shifts
class1 = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))
class2 = np.random.normal(loc=3, scale=1, size=(n_samples, n_features))
class3 = np.random.normal(loc=6, scale=1, size=(n_samples, n_features))

# Combine into full dataset
X = np.vstack([class1, class2, class3])
y_true = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)  # true labels



# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA result
plt.figure(figsize=(8, 6))
for label in np.unique(y_true):
    plt.scatter(X_pca[y_true == label, 0], X_pca[y_true == label, 1], label=f'Class {label}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: First Two Principal Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# K-means clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Compare true labels with clustering results
conf_matrix = pd.crosstab(y_true, clusters, rownames=['Actual'], colnames=['Clustered'])
print(conf_matrix)
