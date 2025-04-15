import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load NCI Gene Expression Data
def load_data(data):
    df=pd.read_csv(data)
    print(df.head())
    print(df.dtypes)

    X=df.drop(['Unnamed: 0'],axis=1)
    y=df['Unnamed: 0']

    return X, y


# 2. Preprocess the data
def preprocess_data(X):
    # Standardize the dataset (mean = 0, variance = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# 3. PCA Dimensionality Reduction
def pca_reduction(X, n_components=50):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance ratio (PCA): {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_pca


# 4. Hierarchical Clustering for Feature Reduction
def hierarchical_clustering(X, n_clusters=50):
    agg_clust = AgglomerativeClustering(n_clusters=n_clusters)
    # Fit the hierarchical clustering model
    cluster_labels = agg_clust.fit_predict(X.T)  # Transpose because we are clustering genes (features)
    return cluster_labels


# 5. Build and Evaluate Model (RandomForest)
def evaluate_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 6. Main Function
def main():
    df = "C:/Users/Naga Nandi Reddy/Downloads/NCI60.csv"

    # 1. Load and preprocess data
    X, y = load_data(df)
    X_scaled = preprocess_data(X)

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 3. PCA - Reduce to 10 components
    X_train_pca = pca_reduction(X_train, n_components=20)
    X_test_pca = pca_reduction(X_test, n_components=20)

    # 4. Hierarchical Clustering - Reduce features to 50 clusters
    cluster_labels_train = hierarchical_clustering(X_train, n_clusters=10)
    cluster_labels_test = hierarchical_clustering(X_test, n_clusters=10)

    # For hierarchical clustering, we use the mean expression in each cluster as features
    X_train_clustered = np.array(
        [np.mean(X_train[:, cluster_labels_train == i], axis=1) for i in np.unique(cluster_labels_train)]).T
    X_test_clustered = np.array(
        [np.mean(X_test[:, cluster_labels_test == i], axis=1) for i in np.unique(cluster_labels_test)]).T

    # 5. Train and evaluate models
    print("Evaluating PCA Approach:")
    evaluate_model(X_train_pca, X_test_pca, y_train, y_test)

    print("\nEvaluating Hierarchical Clustering Approach:")
    evaluate_model(X_train_clustered, X_test_clustered, y_train, y_test)


if __name__ == "__main__":
    main()
