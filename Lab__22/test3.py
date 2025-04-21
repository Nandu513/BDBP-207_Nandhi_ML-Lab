import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, roc_auc_score
)
from ISLP import load_data
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# def split_data(df_path):
#     df1 = load_data(df_path)
#     X = df1['data']
#     y = df1['labels']
#     print(X.shape)
#
#     plt.clf()
#     y.groupby('label').size().plot(kind='bar')
#     plt.title('Distribution of Labels')
#     plt.xlabel('Label')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
#
#     y_encoded = LabelEncoder().fit_transform(y['label'])
#
#     sc = StandardScaler()
#     x_scaled = sc.fit_transform(X)
#
#     return train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=42)

def split_data(df_path):
    df1 = load_data(df_path)
    X = df1['data']
    y = df1['labels']

    # Filter out labels with frequency < 2
    label_counts = y['label'].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    mask = y['label'].isin(valid_labels)
    X = X[mask]
    y = y[mask]

    print(X.shape)

    # Plot label distribution after filtering
    plt.clf()
    y.groupby('label').size().plot(kind='bar')
    plt.title('Distribution of Labels (Filtered)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Encode labels
    y_encoded = LabelEncoder().fit_transform(y['label'])

    # Standardize features
    sc = StandardScaler()
    x_scaled = sc.fit_transform(X)

    return train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=42)

def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # print("\nTRAINING RESULTS:\n===============================")
    # print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    # print("Classification Report (Train):")
    # print(classification_report(y_train, y_train_pred))
    #
    # print("\nTESTING RESULTS:\n===============================")
    # print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    # print("Classification Report (Test):")
    # print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    # cm = confusion_matrix(y_test, y_test_pred)
    # group_names = ['True Neg (TN)', 'False Pos (FP)', 'False Neg (FN)', 'True Pos (TP)']
    # group_counts = [f"{value}" for value in cm.flatten()]
    # labels = [f"{name}\nCount: {count}" for name, count in zip(group_names, group_counts)]
    # labels = np.asarray(labels).reshape(2, 2)
    #
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    # plt.title("Confusion Matrix (Test Set)")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.tight_layout()
    # plt.show()
    #
    # # ROC AUC Score (for binary classification)
    # if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
    #     y_test_proba = model.predict_proba(x_test)[:, 1]
    #     roc_auc = roc_auc_score(y_test, y_test_proba)
    #     print(f"ROC AUC Score (Test): {roc_auc:.4f}")

    # Feature Importance Plot
    # importances = model.feature_importances_
    # feat_df = pd.DataFrame({
    #     'Feature': x_train.columns,
    #     'Importance': importances
    # }).sort_values(by='Importance', ascending=False)
    #
    # plt.figure(figsize=(8, 6))
    # sns.barplot(data=feat_df, x='Importance', y='Feature')
    # plt.title("Feature Importances - Random Forest Classifier")
    # plt.tight_layout()
    # plt.show()

    # Feature usage count across all trees
    # feature_counts = Counter()
    # for tree in model.estimators_:
    #     used = tree.tree_.feature[tree.tree_.feature >= 0]
    #     feature_counts.update(used)
    #
    # print("\nFeature Usage Count in All Trees:")
    # for idx, count in feature_counts.items():
    #     print(f"Feature '{x_train.columns[idx]}' was used in {count} splits")


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
    df = "NCI60"
    x_train, x_test, y_train, y_test = split_data(df)
    random_forest(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
