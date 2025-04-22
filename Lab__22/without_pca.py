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

def split_data(df_path):
    df1 = load_data(df_path)
    X = df1['data']
    y = df1['labels']

    # Filter out labels with frequency < 2
    # label_counts = y['label'].value_counts()
    # valid_labels = label_counts[label_counts >= 2].index
    # mask = y['label'].isin(valid_labels)
    # X = X[mask]
    # y = y[mask]

    print(X.shape)

    # Plot label distribution after filtering
    plt.clf()
    y.groupby('label').size().plot(kind='bar')
    plt.title('Distribution of Labels (Filtered)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()

    # Encode labels
    y_encoded = LabelEncoder().fit_transform(y['label'])

    # Standardize features
    sc = StandardScaler()
    x_scaled = sc.fit_transform(X)

    return train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=42)

def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

def random_forest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42, criterion='entropy')

    param_grid = {
        'n_estimators': [10,20,50,100],
        'max_depth': [5, 7, 10, None],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.001, 0.01]
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Save model
    # joblib.dump(best_model, "random_forest_classifier.pkl")
    # print("\nModel saved as 'random_forest_classifier.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'Random Forest Classifier': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

def logistic_pipeline(x_train, x_test, y_train, y_test):

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1'],
        'solver': ['liblinear','newton-cg','saga']
    }

    grid = GridSearchCV(LogisticRegression(max_iter=100000), param_grid, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)

    model = grid.best_estimator_

    evaluate(model, x_train, x_test, y_train, y_test)

    scores = {'Logistic reg':

        {
            'Train Accuracy': accuracy_score(y_train, model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, model.predict(x_test)),
        }}

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

def adaboost(x_train, x_test, y_train, y_test):
    ada_pipeline = Pipeline([
        ('ada', AdaBoostClassifier(random_state=42))
    ])

    ada_param_grid = {
        'ada__n_estimators': [50, 100, 150],
        'ada__learning_rate': [0.5, 1.0, 1.5]
    }

    ada_grid = GridSearchCV(ada_pipeline, ada_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    ada_grid.fit(x_train, y_train)

    print("\n[AdaBoost] Best Parameters Found:")
    print(ada_grid.best_params_)

    ada_best_model = ada_grid.best_estimator_
    # joblib.dump(ada_best_model, "adaboost_pca_classifier.pkl")
    # print("AdaBoost model saved as 'adaboost_pca_classifier.pkl'")

    evaluate(ada_best_model, x_train, x_test, y_train, y_test)

    scores = {'AdaBoost':

    {
        'Train Accuracy': accuracy_score(y_train, ada_best_model.predict(x_train)),
        'Test Accuracy': accuracy_score(y_test, ada_best_model.predict(x_test)),
    }}

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

def main():
    df = "NCI60"
    x_train, x_test, y_train, y_test = split_data(df)
    random_forest(x_train, x_test, y_train, y_test)
    adaboost(x_train, x_test, y_train, y_test)
    logistic_pipeline(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()

