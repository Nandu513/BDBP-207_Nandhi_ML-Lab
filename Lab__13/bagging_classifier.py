import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, roc_auc_score
)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print("\nTRAINING RESULTS: \n===============================")
    clf_report_train = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report_train}")

    print("\nTESTING RESULTS: \n===============================")
    clf_report_test = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report_test}")


    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Define label names
    group_names = ['True Neg (TN)', 'False Pos (FP)', 'False Neg (FN)', 'True Pos (TP)']
    group_counts = [f"{value}" for value in cm.flatten()]
    labels = [f"{name}\nCount: {count}" for name, count in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    # Plot heatmap with labels
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.title("Test Set Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # ROC AUC Score
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        y_test_proba = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC AUC Score (Test): {roc_auc:.4f}")

    # Feature Importance from ensemble (average)
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        importances = np.mean(
            [tree.feature_importances_ for tree in model.estimators_],
            axis=0
        )

        # Check if shape matches feature count
        if len(importances) == x_train.shape[1]:
            feat_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(8, 6))
            sns.barplot(data=feat_df, x='Importance', y='Feature')
            plt.title("Average Feature Importances (Bagging Ensemble)")
            plt.tight_layout()
            plt.show()
        else:
            print("Skipping feature importance plot (shape mismatch).")


def load_data(df_path):
    df = pd.read_csv(df_path)
    x = df.drop(columns=['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test


def bagging(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    bagging_clf = BaggingClassifier(estimator=tree, random_state=99)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'estimator__max_depth': [3, 5, 7, None],
        'max_samples': [0.8, 1.0],
        'max_features': [0.8, 1.0]
    }

    grid = GridSearchCV(bagging_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Save the best model
    joblib.dump(best_model, "bagging_best_model.pkl")
    print("\nModel saved as 'bagging_best_model.pkl'")

    # Load the saved model
    # loaded_model = joblib.load("bagging_best_model.pkl")

    # Now you can use the model to make predictions
    # predictions = loaded_model.predict(x_test)

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'Bagging Classifier': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        },
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/diabetes.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    bagging(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
