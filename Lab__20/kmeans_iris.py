import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from Kmeans import Kmeans


def load_and_preprocess_iris():
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_true


def evaluate_clustering(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")


def plot_clusters(X, y_pred, y_true):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(12, 5))

    # Predicted Clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', s=50)
    plt.title("KMeans Cluster Assignments")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Ground Truth Labels
    plt.subplot(1, 2, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', s=50)
    plt.title("True Iris Labels")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.tight_layout()
    plt.show()


def main():
    # 1. Load and preprocess the dataset
    X_scaled, y_true = load_and_preprocess_iris()

    # 2. Apply KMeans clustering
    kmeans = Kmeans(K=3, tolerance=1e-6, iterations=20)
    centroids, y_pred = kmeans.fit(X_scaled)

    # 3. Evaluate the clustering performance
    evaluate_clustering(y_true, y_pred)

    # 4. Plot the clusters
    plot_clusters(X_scaled, y_pred, y_true)


if __name__ == "__main__":
    main()
