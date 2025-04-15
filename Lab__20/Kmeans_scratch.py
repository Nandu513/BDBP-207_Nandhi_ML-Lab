import numpy as np

class Kmeans:
    """
    K-Means clustering algorithm implementation.

    Parameters:
    K (int): Number of clusters
    tolerance (float): Threshold to determine convergence (default: small value like 1e-4)
    iterations (int): Maximum number of iterations for convergence

    Attributes:
        K (int): Number of clusters
        centroids (numpy.ndarray): Array containing the centroids of each cluster

    Methods:
        __init__(self, K): Initializes the Kmeans instance with the specified number of clusters.
        initialize_centroids(self, X): Initializes the centroids for each cluster by selecting K random points from the dataset.
        assign_points_centroids(self, X): Assigns each point in the dataset to the nearest centroid.
        compute_mean(self, X, points): Computes the mean of the points assigned to each centroid.
        fit(self, X, iterations=10): Clusters the dataset using the K-Means algorithm.
    """

    def __init__(self, K, tolerance, iterations):
        assert K > 0, "K should be a positive integer."
        self.K = K
        self.tolerance = tolerance
        self.iterations = iterations

    def initialize_centroids(self, X):
        assert X.shape[0] >= self.K, "Number of data points should be greater than or equal to K."

        randomized_X = np.random.permutation(X.shape[0])
        centroid_idx = randomized_X[:self.K]  # get the indices for the centroids
        self.centroids = X[centroid_idx]  # assign the centroids to the selected points

    def assign_points_centroids(self, X):
        """
        Assign each point in the dataset to the nearest centroid.

        Parameters:
        X (numpy.ndarray): dataset to cluster

        Returns:
        numpy.ndarray: array containing the index of the centroid for each point
        """
        X = np.expand_dims(X, axis=1)  # expand dimensions to match shape of centroids
        distance = np.linalg.norm((X - self.centroids),
                                  axis=-1)  # calculate Euclidean distance between each point and each centroid
        points = np.argmin(distance, axis=1)  # assign each point to the closest centroid
        assert len(points) == X.shape[0], "Number of assigned points should equal the number of data points."
        return points

    def compute_mean(self, X, points):
        """
        Compute the mean of the points assigned to each centroid.

        Parameters:
        X (numpy.ndarray): dataset to cluster
        points (numpy.ndarray): array containing the index of the centroid for each point

        Returns:
        numpy.ndarray: array containing the new centroids for each cluster
        """
        centroids = np.zeros((self.K, X.shape[1]))  # initialize array to store centroids
        for i in range(self.K):
            centroid_mean = X[points == i].mean(axis=0)  # calculate mean of the points assigned to the current centroid
            centroids[i] = centroid_mean  # assign the new centroid to the mean of its points
        return centroids

    def fit(self, X):
        """
        Cluster the dataset using the K-Means algorithm.

        Parameters:
        X (numpy.ndarray): dataset to cluster
        iterations (int): number of iterations to perform (default=10)

        Returns:
        numpy.ndarray: array containing the final centroids for each cluster
        numpy.ndarray: array containing the index of the centroid for each point
        """
        self.initialize_centroids(X)  # initialize the centroids
        errors = []
        for i in range(self.iterations):
            points = self.assign_points_centroids(X)  # assign each point to the nearest centroid
            new_centroids = self.compute_mean(X, points)  # compute the new centroids based on the mean of their points

            # Track movement (cost = sum of distances of points to their centroid)
            cost = 0
            for j in range(self.K):
                cluster_points = X[points == j]
                if len(cluster_points) > 0:
                    cost += np.sum((cluster_points - new_centroids[j]) ** 2)
            errors.append(cost)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tolerance):
                print(f"Converged at iteration {i}")
                break
            self.centroids = new_centroids

            # Assertions for debugging and validation
            assert len(self.centroids) == self.K, "Number of centroids should equal K."
            assert X.shape[1] == self.centroids.shape[1], "Dimensionality of centroids should match input data."
            assert max(points) < self.K, "Cluster index should be less than K."
            assert min(points) >= 0, "Cluster index should be non-negative."

        return self.centroids, points, errors


    def predict(self, X):
        """
        Assign new points to the nearest cluster after training.
        """
        X = np.expand_dims(X, axis=1)
        distance = np.linalg.norm(X - self.centroids, axis=-1)
        return np.argmin(distance, axis=1)
