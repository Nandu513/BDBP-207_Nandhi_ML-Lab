import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print("\nTRAINING RESULTS:\n===============================")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_train, y_train_pred))

    print("\nTESTING RESULTS:\n===============================")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    # Confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Multiclass XGBoost")
    plt.tight_layout()
    plt.show()

    # Feature importances
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature')
    plt.title("Feature Importances - Multiclass XGBoost")
    plt.tight_layout()
    plt.show()


def load_data(df_path):
    df = pd.read_csv(df_path)

    # Encode string labels if needed
    if df['species'].dtype == 'object':
        le = LabelEncoder()
        df['species'] = le.fit_transform(df['species'])

    x = df.drop(columns=['species'], axis=1)
    y = df['species']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    return x_train, x_test, y_train, y_test, len(np.unique(y))


def xgboost_multiclass(x_train, x_test, y_train, y_test, num_classes):
    xgb = XGBClassifier(
        objective='multi:softprob',  # for multiclass probability output
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42
    )

    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    joblib.dump(best_model, "xgboost_multiclass_classifier.pkl")
    print("\nModel saved as 'xgboost_multiclass_classifier.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'XGBoost Multiclass': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/iris.csv"
    x_train, x_test, y_train, y_test, num_classes = load_data(df_path)
    xgboost_multiclass(x_train, x_test, y_train, y_test, num_classes)


if __name__ == "__main__":
    main()
