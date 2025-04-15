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

def plot_clusters_3D(X, y_pred, centroids):
    from mpl_toolkits.mplot3d import Axes3D
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)
    centroids_3d = pca.transform(centroids)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_pred, cmap='viridis', s=50, alpha=0.6)
    ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', marker='X', s=200, label='Centroids')
    ax.set_title("KMeans Clusters (3D PCA View)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.legend()
    plt.show()

def main():
    # 1. Load and preprocess the dataset
    X_scaled, y_true = load_and_preprocess_iris()

    # 2. Apply KMeans clustering
    kmeans = Kmeans(K=3, tolerance=1e-6, iterations=20)
    centroids, y_pred, errors = kmeans.fit(X_scaled)

    # 3. Evaluate the clustering performance
    evaluate_clustering(y_true, y_pred)

    # 4. Plot the clusters
    plot_clusters(X_scaled, y_pred, y_true)

    # 5. Plot in 3D
    plot_clusters_3D(X_scaled, y_pred, centroids)

    # 6. Plot convergence graph
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.title("KMeans Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Sum of Squared Distances (Cost)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
