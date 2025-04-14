import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print("\nTRAINING RESULTS:\n===============================")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("Classification Report (Train):")
    print(classification_report(y_train, y_train_pred))

    print("\nTESTING RESULTS:\n===============================")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def load_data(df_path):
    df = pd.read_csv(df_path)
    print(df.head())
    x = df.drop(columns=['species'])
    y = df['species']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test


def adaboost_classifier(x_train, x_test, y_train, y_test):
    base_tree = DecisionTreeClassifier(random_state=42)
    ada = AdaBoostClassifier(estimator=base_tree, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1.0],
        'estimator__max_depth': [1, 2, 3]
    }

    grid = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    print("\nClass to Integer Mapping:")
    for idx, class_label in enumerate(best_model.classes_):
        print(f"Class {class_label} -> Integer {idx}")

    # Save model
    joblib.dump(best_model, "adaboost_classifier.pkl")
    print("\nModel saved as 'adaboost_classifier.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'AdaBoost Classifier': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/iris.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    adaboost_classifier(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
