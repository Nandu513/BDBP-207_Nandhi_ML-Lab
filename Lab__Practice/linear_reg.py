import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set the seed for reproducibility
np.random.seed(1)

# (a) Generate 200 observations for feature X from N(0, 1)
X = np.random.normal(loc=0, scale=1, size=200)

# (b) Generate 200 observations for noise e from N(0, 0.25)
e = np.random.normal(loc=0, scale=np.sqrt(0.25), size=200)

# (c) Generate y using the model y = -1.1 + 0.6*X + e
y = -1.1 + 0.6 * X + e

# Comment:
# The length of the vector y is 200
# The true values of theta_0 and theta_1 in this model are:
# theta_0 = -1.1 (intercept)
# theta_1 = 0.6  (coefficient for X)

# (d) Create a scatter plot of X vs y
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.7)
plt.title("Scatter Plot of X vs y")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Comment: The scatter plot shows a linear trend with noise around the line.

# (e) 70-30 train-test split
X = X.reshape(-1, 1)  # Reshape for sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Fit least squares linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Scatter plot of predictions vs test features
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual y_test')
plt.plot(X_test, y_pred, color='red', label='Predicted y', linewidth=2)
plt.title("X_test vs y_test with Regression Line")
plt.xlabel("X_test")
plt.ylabel("y_test and y_pred")
plt.legend()
plt.grid(True)
plt.show()
