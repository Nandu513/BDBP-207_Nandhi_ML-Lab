import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from kneed import KneeLocator


# 1. Load and preprocess the dataset
def load_and_preprocess_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def plot_elbow(X_scaled, k_range):
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()

    return inertia

# 2. KMeans + Local Regression
def kmeans_local_regression(X_train, y_train, K):
    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)

    regressors = []
    for i in range(K):
        X_cluster = X_train[cluster_labels == i]
        y_cluster = y_train[cluster_labels == i]

        model = LinearRegression()
        model.fit(X_cluster, y_cluster)
        regressors.append(model)

    return kmeans, regressors

# 3. Predict using closest cluster's regressor
def predict_with_local_models(X, kmeans, regressors):
    preds = []
    for x in X:
        cluster = kmeans.predict([x])[0]
        pred = regressors[cluster].predict([x])[0]
        preds.append(pred)
    return np.array(preds)


# 4. Evaluation + Plotting
def evaluate_and_plot(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title("KMeans Regression: True vs Predicted")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()


# 5. Main Function
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    k_values = range(1, 11)
    inertias=plot_elbow(X_train, k_values)


    # Use KneeLocator to find the elbow point
    knee = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
    optimal_k = knee.knee
    print(f"Optimal number of clusters found: {optimal_k}")

    K = 5  # You can try different values here
    kmeans, regressors = kmeans_local_regression(X_train, y_train, K)

    y_pred = predict_with_local_models(X_test, kmeans, regressors)

    evaluate_and_plot(y_test, y_pred)


if __name__ == "__main__":
    main()
