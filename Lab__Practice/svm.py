import pandas as pd
import numpy as np
from ISLP import load_data
from sklearn.model_selection import train_test_split

# Load data
OJ = load_data('OJ')

# Encode target variable
OJ['Purchase'] = (OJ['Purchase'] == 'CH').astype(int)  # CH=1, MM=0

# Train-test split
train_OJ = OJ.sample(n=1000, random_state=42)
test_OJ = OJ.drop(train_OJ.index)


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Separate features and target
X_train = train_OJ.drop(columns='Purchase')
y_train = train_OJ['Purchase']
X_test = test_OJ.drop(columns='Purchase')
y_test = test_OJ['Purchase']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=np.number))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=np.number))

# Fit LinearSVC
model = LinearSVC(C=0.01, max_iter=10000, dual=False)
model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


from sklearn.model_selection import GridSearchCV

param_grid = {'C': np.logspace(-2, 1, 10)}
grid = GridSearchCV(LinearSVC(max_iter=10000, dual=False), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print("Best C:", grid.best_params_['C'])


best_model = grid.best_estimator_

train_error = 1 - accuracy_score(y_train, best_model.predict(X_train_scaled))
test_error = 1 - accuracy_score(y_test, best_model.predict(X_test_scaled))
print(f"Train Error: {train_error:.4f}")
print(f"Test Error: {test_error:.4f}")

from sklearn.svm import SVC

# Fit RBF SVM
rbf_model = SVC(kernel='rbf')  # gamma='scale' by default
rbf_model.fit(X_train_scaled, y_train)

# Evaluate
train_rbf_acc = accuracy_score(y_train, rbf_model.predict(X_train_scaled))
test_rbf_acc = accuracy_score(y_test, rbf_model.predict(X_test_scaled))
print(f"RBF SVM Train Accuracy: {train_rbf_acc:.4f}")
print(f"RBF SVM Test Accuracy: {test_rbf_acc:.4f}")


import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the Weekly dataset
Weekly = load_data('Weekly')

# Encode the 'Direction' column (Up/Down) into binary labels
le = LabelEncoder()
Weekly['Direction'] = le.fit_transform(Weekly['Direction'])  # Up = 1, Down = 0

# Extract features and target
X = Weekly[['Lag1', 'Lag2']].values
y = Weekly['Direction'].values
n = len(y)

# Initialize list to store accuracy for each observation
accuracies = []

# LOOCV loop
for i in range(n):
    # Create train and test sets
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X[i].reshape(1, -1)
    y_test = y[i]

    # Fit logistic regression model
    model = LogisticRegression(solver='liblinear')  # solver for small datasets
    model.fit(X_train, y_train)

    # Predict on the test point
    y_pred = model.predict(X_test)[0]

    # Store whether prediction was correct
    accuracies.append(y_pred == y_test)

# Calculate average accuracy
loocv_accuracy = np.mean(accuracies)
print(f"LOOCV Estimated Test Accuracy: {loocv_accuracy:.4f}")


