from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, roc_auc_score
)
from ISLP import load_data
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut


def split_data(df_path, k=50):
    df1 = load_data(df_path)
    X = df1['data']
    y = df1['labels']

    print("Original shape:", X.shape)

    # Filter out labels with frequency < 2
    label_counts = y['label'].value_counts()
    valid_labels = label_counts[label_counts >= 3].index
    mask = y['label'].isin(valid_labels)
    X = X[mask]
    y = y[mask]

    print("After filtering: ",X.shape)

    # Encode labels for use in feature selection
    y_encoded = LabelEncoder().fit_transform(y['label'])

    # Feature Selection: SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    X_kbest = selector.fit_transform(X, y_encoded)
    print(f"After SelectKBest (top {k} features):", X_kbest.shape)

    sel = VarianceThreshold(threshold=0.20)  # Removes low-variance features
    X_var = sel.fit_transform(X_kbest)
    print("After Variance Thresholding:", X_var.shape)

    rf = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=50)
    rf = rf.fit(X_var, y)
    #rf.feature_importances_
    model = SelectFromModel(rf, prefit=True)
    X_new = model.transform(X_var)
    print("Tree based selection:",X_new.shape)

    # Standardize features
    sc = StandardScaler()
    x_scaled = sc.fit_transform(X_new)

    return train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=99,stratify=y_encoded)


def random_forest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42, criterion='entropy')

    param_grid = {
        'n_estimators': [10,20,50,100],
        'max_depth': [4,5, 7, 10, None],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.001, 0.01]
    }

    grid = GridSearchCV(rf, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    print("Mean cross-validated score of the best_estimator: ",grid.best_score_)

    # Get index of the best model
    best_index = grid.best_index_
    # Get the std of the best model's CV scores
    std_score = grid.cv_results_['std_test_score'][best_index]
    print(f"Standard Deviation of CV scores for Best Estimator: {std_score:.4f}")

    # Save model
    # joblib.dump(best_model, "random_forest_classifier.pkl")
    # print("\nModel saved as 'random_forest_classifier.pkl'")

    scores = {
        'Random Forest Classifier': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

def rf_pca(x_train, x_test, y_train, y_test):
    pipeline = Pipeline([
        ('pca', PCA()),
        ('rf', RandomForestClassifier(random_state=99, criterion='entropy'))
    ])

    param_grid = {
        'pca__n_components': [20,25,32],
        'rf__n_estimators': [50,100,150,200],
        'rf__max_depth': [4,5,6,7,8,10],
        'rf__max_features': ['log2'],
        'rf__ccp_alpha': [0.001,0,0.1,0.01]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    print("Mean cross-validated score of the best_estimator: ",grid.best_score_)

    # Get index of the best model
    best_index = grid.best_index_
    # Get the std of the best model's CV scores
    std_score = grid.cv_results_['std_test_score'][best_index]
    print(f"Standard Deviation of CV scores for Best Estimator: {std_score:.4f}")

    # Save model
    # joblib.dump(best_model, "random_forest_pca_classifier.pkl")
    # print("\nModel saved as 'random_forest_pca_classifier.pkl'")


    scores = {
        'Random Forest + PCA': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def svc_classifier_pca(x_train, x_test, y_train, y_test):
    pipeline = Pipeline([
        ('pca', PCA()),
        ('svc', SVC(probability=True))
    ])

    param_grid = {
        'pca__n_components': [20, 25, 30],  # adjust as needed
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\n[SVC] Best Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    print("Mean cross-validated score of the best_estimator: ",grid.best_score_)

    # Get index of the best model
    best_index = grid.best_index_
    # Get the std of the best model's CV scores
    std_score = grid.cv_results_['std_test_score'][best_index]
    print(f"Standard Deviation of CV scores for Best Estimator: {std_score:.4f}")

    scores = {
        'SVC + PCA': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

    # Optional: Save model
    # joblib.dump(best_model, "svc_pca_classifier.pkl")
    # print("\nModel saved as 'svc_pca_classifier.pkl'")

def svc_classifier_no_pca(x_train, x_test, y_train, y_test):
    svc = SVC(probability=True)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(svc, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\n[SVC No PCA] Best Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    print("Mean cross-validated score of the best_estimator: ",grid.best_score_)

    # Get index of the best model
    best_index = grid.best_index_
    # Get the std of the best model's CV scores
    std_score = grid.cv_results_['std_test_score'][best_index]
    print(f"Standard Deviation of CV scores for Best Estimator: {std_score:.4f}")

    scores = {
        'SVC (No PCA)': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

    # Optional: Save model
    # joblib.dump(best_model, "svc_classifier_no_pca.pkl")
    # print("\nModel saved as 'svc_classifier_no_pca.pkl'")


def main():
    df = "NCI60"
    # x_train, x_test, y_train, y_test = split_data(df)
    x_train, x_test, y_train, y_test = split_data(df, k=2000)
    random_forest(x_train, x_test, y_train, y_test)
    rf_pca(x_train, x_test, y_train, y_test)
    svc_classifier_pca(x_train, x_test, y_train, y_test)
    svc_classifier_no_pca(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()


# /home/ibab/PycharmProjects/PythonProject/.venv/bin/python /home/ibab/Desktop/BDBP-207_Nandhi_ML-Lab/Lab__22/feature select.py
# Original shape: (64, 6830)
# After filtering:  (57, 6830)
# After SelectKBest (top 2000 features): (57, 2000)
# After Variance Thresholding: (57, 1771)
#
# Best Parameters Found:
# {'ccp_alpha': 0.0, 'max_depth': 5, 'max_features': 'log2', 'n_estimators': 50}
#
# SCORES SUMMARY:
#                           Train Accuracy  Test Accuracy
# Random Forest Classifier             1.0       0.333333
#
# Best Parameters Found:
# {'pca__n_components': 20, 'rf__ccp_alpha': 0.001, 'rf__max_depth': 5, 'rf__max_features': 'log2', 'rf__n_estimators': 150}
#
# SCORES SUMMARY:
#                      Train Accuracy  Test Accuracy
# Random Forest + PCA             1.0            0.5
#
# [SVC] Best Parameters Found:
# {'pca__n_components': 30, 'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}
#
# SCORES SUMMARY:
#            Train Accuracy  Test Accuracy
# SVC + PCA             1.0       0.777778
#
# [SVC No PCA] Best Parameters Found:
# {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
#
# SCORES SUMMARY:
#               Train Accuracy  Test Accuracy
# SVC (No PCA)             1.0       0.833333
#
# Process finished with exit code 0


# /home/ibab/PycharmProjects/PythonProject/.venv/bin/python /home/ibab/Desktop/BDBP-207_Nandhi_ML-Lab/Lab__22/feature select.py
# Original shape: (64, 6830)
# After filtering:  (57, 6830)
# After SelectKBest (top 2000 features): (57, 2000)
# After Variance Thresholding: (57, 1771)
# Tree based selection: (57, 508)
#
# Best Parameters Found:
# {'ccp_alpha': 0.0, 'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}
#
# SCORES SUMMARY:
#                           Train Accuracy  Test Accuracy
# Random Forest Classifier             1.0            0.5
#
# Best Parameters Found:
# {'pca__n_components': 20, 'rf__ccp_alpha': 0.001, 'rf__max_depth': 4, 'rf__max_features': 'log2', 'rf__n_estimators': 150}
#
# SCORES SUMMARY:
#                      Train Accuracy  Test Accuracy
# Random Forest + PCA             1.0       0.722222
#
# [SVC] Best Parameters Found:
# {'pca__n_components': 20, 'svc__C': 0.1, 'svc__gamma': 'scale', 'svc__kernel': 'linear'}
#
# SCORES SUMMARY:
#            Train Accuracy  Test Accuracy
# SVC + PCA             1.0       0.944444
#
# [SVC No PCA] Best Parameters Found:
# {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
#
# SCORES SUMMARY:
#               Train Accuracy  Test Accuracy
# SVC (No PCA)             1.0       0.833333
#
# Process finished with exit code 0


# /home/ibab/PycharmProjects/PythonProject/.venv/bin/python /home/ibab/Desktop/BDBP-207_Nandhi_ML-Lab/Lab__22/feature select.py
# Original shape: (64, 6830)
# After filtering:  (57, 6830)
# After SelectKBest (top 2000 features): (57, 2000)
# After Variance Thresholding: (57, 1771)
# Tree based selection: (57, 384)
#
# Best Parameters Found:
# {'ccp_alpha': 0.0, 'max_depth': 5, 'max_features': 'log2', 'n_estimators': 100}
# Mean cross-validated score of the best_estimator:  0.6666666666666666
# Standard Deviation of CV scores for Best Estimator: 0.4714
#
# SCORES SUMMARY:
#                           Train Accuracy  Test Accuracy
# Random Forest Classifier             1.0       0.666667
#
# Best Parameters Found:
# {'pca__n_components': 20, 'rf__ccp_alpha': 0.001, 'rf__max_depth': 4, 'rf__max_features': 'log2', 'rf__n_estimators': 150}
# Mean cross-validated score of the best_estimator:  0.717948717948718
# Standard Deviation of CV scores for Best Estimator: 0.4500
#
# SCORES SUMMARY:
#                      Train Accuracy  Test Accuracy
# Random Forest + PCA             1.0            1.0
#
# [SVC] Best Parameters Found:
# {'pca__n_components': 30, 'svc__C': 0.1, 'svc__gamma': 'scale', 'svc__kernel': 'linear'}
# Mean cross-validated score of the best_estimator:  0.7435897435897436
# Standard Deviation of CV scores for Best Estimator: 0.4367
#
# SCORES SUMMARY:
#            Train Accuracy  Test Accuracy
# SVC + PCA             1.0       0.888889
#
# [SVC No PCA] Best Parameters Found:
# {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
# Mean cross-validated score of the best_estimator:  0.717948717948718
# Standard Deviation of CV scores for Best Estimator: 0.4500
#
# SCORES SUMMARY:
#               Train Accuracy  Test Accuracy
# SVC (No PCA)             1.0       0.888889
#
# Process finished with exit code 0
