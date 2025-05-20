# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import SGDClassifier
# from ISLP import load_data
# from skopt import BayesSearchCV
# from skopt.space import Real
# from sklearn.metrics import make_scorer, accuracy_score
#
# def find_best_hyperparameters(X_train, y_train):
#     # Define search space
#     search_space = {
#                 'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
#                 'eta0': Real(1e-4, 1e-1, prior='log-uniform'),
#             }
#
#     # Define model
#     clf = SGDClassifier(loss='log_loss', penalty='elasticnet',
#                         l1_ratio=0.5, learning_rate='constant',
#                         max_iter=1000, random_state=42)
#
#     # Bayesian Optimization via BayesSearchCV
#     opt = BayesSearchCV(
#         estimator=clf,
#         search_spaces=search_space,
#         n_iter=25,  # You can increase for better search
#         scoring=make_scorer(accuracy_score),
#         cv=KFold(n_splits=5, shuffle=True, random_state=42),
#         n_jobs=-1,
#         random_state=42
#     )
#
#     # Fit optimizer
#     opt.fit(X_train, y_train)
#
#     best_lambda = opt.best_params_['alpha']
#     best_eta0 = opt.best_params_['eta0']
#
#     print(f"Best Hyperparameters from Bayesian Optimization: lambda = {best_lambda}, eta0 = {best_eta0}")
#     return best_lambda, best_eta0
#
#
#
# def load_smarket_data():
#     data = load_data('Smarket')
#     print(data.head())
#
#     # Convert 'Direction' to binary: Up = 1, Down = 0
#     data['Direction'] = data['Direction'].map({'Up': 1, 'Down': 0})
#
#     # Use only lag variables and volume as features
#     features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
#     X = data[features].values
#     y = data['Direction'].values
#
#     # Train-test split (2001–2004 train, 2005 test)
#     train_idx = data['Year'] < 2005
#     test_idx = data['Year'] == 2005
#
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#
#     return X_train, X_test, y_train, y_test
#
# def logistic_regression_smarket():
#     X_train, X_test, y_train, y_test = load_smarket_data()
#
#     sc = StandardScaler()
#     X_train_scaled = sc.fit_transform(X_train)
#     X_test_scaled = sc.transform(X_test)
#
#     best_lambda, best_eta0 = find_best_hyperparameters(X_train_scaled, y_train)
#
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
#     best_clf = None
#
#     for i, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
#         # print(f"Fold {i}:")
#         x_tr, x_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
#         y_tr, y_val = y_train[train_idx], y_train[val_idx]
#
#         clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=best_lambda,
#                             l1_ratio=0.5, learning_rate='constant', eta0=best_eta0,
#                             max_iter=1000, random_state=42)
#         clf.fit(x_tr, y_tr)
#         y_pred = clf.predict(x_val)
#         acc = accuracy_score(y_val, y_pred)
#         scores.append(acc)
#
#         if i == 4:
#             best_clf = clf
#
#     print(f"Mean Accuracy: {np.mean(scores)}")
#     print(f"Standard Deviation: {np.std(scores)}")
#
#     y_test_pred = best_clf.predict(X_test_scaled)
#     test_acc = accuracy_score(y_test, y_test_pred)
#     print(f"Test Accuracy: {test_acc}")
#
#
# def main():
#     logistic_regression_smarket()
#
#
# if __name__ == '__main__':
#     main()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_curve
from ISLP import load_data

##pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real

def load_smarket_data():
    data = load_data('Smarket')
    print(data.head())
    # print(data.describe())

    data['Direction'] = data['Direction'].map({'Up': 1, 'Down': 0})
    features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
    X = data[features].values
    y = data['Direction'].values

    train_idx = data['Year'] < 2005
    test_idx = data['Year'] == 2005

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def bayesian_hyperparameter_tuning(X, y):
    clf = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5, max_iter=1000, random_state=42)

    search_space = {
        'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
        'eta0': Real(1e-4, 1e-1, prior='log-uniform'),
    }

    bayes_search = BayesSearchCV(
        estimator=clf,
        search_spaces=search_space,
        n_iter=25,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    bayes_search.fit(X, y)

    # Extract mean and std dev of best result
    best_idx = bayes_search.best_index_
    mean_score = bayes_search.cv_results_['mean_test_score'][best_idx]
    std_score = bayes_search.cv_results_['std_test_score'][best_idx]

    print("Best Parameters (Bayesian Optimization):", bayes_search.best_params_)
    print(f"Best Cross-Validated Accuracy: {mean_score:.4f}")
    print(f"Standard Deviation of CV Accuracy: {std_score:.4f}")

    return bayes_search.best_estimator_

def manual_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def evaluate_model(y_true, y_pred, y_proba):
    tp, tn, fp, fn = manual_confusion_matrix(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Manual AUC via trapezoidal rule on ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = np.trapz(tpr, fpr)

    print(f"\nManual Evaluation Metrics:")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Sensitivity  : {sensitivity:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {auc:.4f}")

    cm = np.array([[tp, fp],
                   [fn, tn]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 1', 'Predicted 0'],
                yticklabels=['Actual 1', 'Actual 0'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# def grid_search_hyperparameter_tuning(X, y):
#     clf = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5,
#                         max_iter=1000, random_state=42)
#
#     param_grid = {
#         'alpha': [0.0001, 0.001, 0.01, 0.1],   # Regularization strength
#         'eta0': [0.001, 0.01, 0.1]             # Learning rate
#     }
#
#     grid_search = GridSearchCV(
#         estimator=clf,
#         param_grid=param_grid,
#         cv=5,
#         scoring='accuracy',
#         return_train_score=True,
#         n_jobs=-1,
#         verbose=0
#     )
#
#     grid_search.fit(X, y)
#
#     best_idx = grid_search.best_index_
#     mean_score = grid_search.cv_results_['mean_test_score'][best_idx]
#     std_score = grid_search.cv_results_['std_test_score'][best_idx]
#
#     print("Best Parameters (Grid Search):", grid_search.best_params_)
#     print(f"Best Cross-Validated Accuracy: {mean_score:.4f}")
#     print(f"Standard Deviation of CV Accuracy: {std_score:.4f}")
#
#     return grid_search.best_estimator_


def logistic_regression_smarket():
    X_train, X_test, y_train, y_test = load_smarket_data()

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    best_model = bayesian_hyperparameter_tuning(X_train_scaled, y_train)

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc}")

    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    evaluate_model(y_test, y_test_pred, y_test_proba)


def main():
    logistic_regression_smarket()


if __name__ == '__main__':
    main()
