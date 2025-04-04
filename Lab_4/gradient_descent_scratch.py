import numpy as np
import pickle
import matplotlib.pyplot as plt 

class LinearRegression:
    """
    Linear Regression Model with Gradient Descent
    """

    def __init__(self, learning_rate, convergence_tol=1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None

    def initialize_parameters(self, n_features):
        self.W = np.random.randn(n_features) * 0.01
        self.b = 0

    def hypothesis(self, X):
        return np.dot(X, self.W) + self.b

    def compute_cost(self, predictions, y):
        m = len(y)
        return np.sum(np.square(predictions - y)) / (2 * m)

    def compute_derivatives(self, X, y, predictions):
        m = X.shape[0]
        error = predictions - y
        dW = np.dot(X.T, error) / m
        db = np.sum(error) / m
        return dW, db

    def gradient_descent_step(self, dW, db):
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def fit(self, X, y, iterations, plot_cost=True):
        assert isinstance(X, np.ndarray), "X must be a NumPy array"
        assert isinstance(y, np.ndarray), "y must be a NumPy array"
        assert X.shape[0] == y.shape[0], "Mismatch in number of samples"
        assert iterations > 0, "Iterations must be positive"

        self.initialize_parameters(X.shape[1])
        costs = []

        for i in range(iterations):
            predictions = self.hypothesis(X)
            cost = self.compute_cost(predictions, y)
            dW, db = self.compute_derivatives(X, y, predictions)
            self.gradient_descent_step(dW, db)
            costs.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i} | Cost: {cost:.6f}")

            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f"Converged at iteration {i}.")
                break

        if plot_cost:
            plt.figure(figsize=(8, 5))
            plt.plot(range(len(costs)), costs, color="blue")
            plt.title("Cost vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.grid(True)
            plt.show()

    def predict(self, X):
        return self.hypothesis(X)

    def save_model(self, filename):
        model_data = {
            'learning_rate': self.learning_rate,
            'convergence_tol': self.convergence_tol,
            'W': self.W,
            'b': self.b
        }
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
        model = cls(model_data['learning_rate'], model_data['convergence_tol'])
        model.W = model_data['W']
        model.b = model_data['b']
        return model
