import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from ISLP import load_data
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

warnings.filterwarnings("ignore")


# 1. Load dataset
def load_dataset(data):
    df1 = load_data(data)
    X = df1['data']
    y = df1['labels']

    print(X)

    plt.clf()
    y.groupby('label').size().plot(kind='bar')
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return X, y


# 2. Standardize
def preprocess_data(X):
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    return sc, X_scaled


# 3. Custom Estimator
class PCABasedAgglomerative(BaseEstimator, ClusterMixin):
    def __init__(self, n_components=2, n_clusters=3, linkage='ward'):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components)
        X_reduced = self.pca_.fit_transform(X)

        self.clusterer_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage
        )
        self.labels_ = self.clusterer_.fit_predict(X_reduced)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


# 4. Custom scorer
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    if len(np.unique(labels)) < 2:
        return -1  # silhouette not valid for 1 cluster
    return silhouette_score(X, labels)


# 5. Main function with Bayesian optimization
def main():
    df = "NCI60"
    X, y = load_dataset(df)
    sc, X_scaled = preprocess_data(X)

    model = PCABasedAgglomerative()

    # Define search space
    search_space = {
        'n_components': Integer(2, 10),
        'n_clusters': Integer(2, 10),
        'linkage': Categorical(['ward', 'complete', 'average'])
    }

    # Use dummy CV split since we're in unsupervised learning
    fake_cv = [(slice(None), slice(None))]

    optimizer = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        scoring=silhouette_scorer,
        n_iter=30,
        cv=fake_cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    optimizer.fit(X_scaled)

    print("Best Parameters:", optimizer.best_params_)
    print("Best Silhouette Score:", optimizer.best_score_)

    # Visualize best clustering result
    best_model = optimizer.best_estimator_
    X_pca = best_model.pca_.transform(X_scaled)
    labels = best_model.labels_

    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title('Best Clustering Result (Bayesian Optimized)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
