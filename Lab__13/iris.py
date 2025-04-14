import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier


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

    # Confusion Matrix for Multiclass
    cm = confusion_matrix(y_test, y_test_pred)
    class_names = np.unique(y_test)  # Get class names (labels)

    # Prepare labels for confusion matrix
    labels = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j:
                label = f"True Pos ({class_names[i]})\nCount: {cm[i, j]}"
            else:
                label = f"{class_names[j]} Pred as {class_names[i]}\nCount: {cm[i, j]}"
            labels.append(label)

    # Reshape the labels for displaying in heatmap
    labels = np.array(labels).reshape(len(class_names), len(class_names))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Test Set) - Multiclass")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Feature Importance Plot
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature')
    plt.title("Feature Importances - Random Forest Classifier")
    plt.tight_layout()
    plt.show()

    # Feature usage count across all trees
    feature_counts = Counter()
    for tree in model.estimators_:
        used = tree.tree_.feature[tree.tree_.feature >= 0]  # Get feature indices used in the tree
        feature_counts.update(used)

    print("\nFeature Usage Count in All Trees:")
    for idx, count in feature_counts.items():
        print(f"Feature '{x_train.columns[idx]}' was used in {count} splits")


def load_data(df_path):
    df = pd.read_csv(df_path)
    print(df.head())
    x = df.drop(columns=['species'])
    y = df['species']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test


def random_forest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42, criterion='entropy')

    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 7, 10, None],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.001, 0.01],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)


    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Assuming the model is already trained as 'best_model' (after GridSearchCV)
    print("Class to Integer Mapping:")
    for idx, class_label in enumerate(best_model.classes_):
        print(f"Class {class_label} -> Integer {idx}")

    # Save model
    joblib.dump(best_model, "random_forest_classifier.pkl")
    print("\nModel saved as 'random_forest_classifier.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'Random Forest Classifier': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/iris.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    random_forest(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
