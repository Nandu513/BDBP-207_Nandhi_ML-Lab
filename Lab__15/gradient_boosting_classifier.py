import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print("\nTRAINING RESULTS:\n===============================")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_train, y_train_pred))

    print("\nTESTING RESULTS:\n===============================")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()

    # Feature importance plot
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature')
    plt.title("Feature Importances - Gradient Boosting Classifier")
    plt.tight_layout()
    plt.show()


def load_data(df_path):
    df = pd.read_csv(df_path)

    # Encode target if it's categorical
    if df['Direction'].dtype == 'object':
        le = LabelEncoder()
        df['Direction'] = le.fit_transform(df['Direction'])

    x = df.drop(columns=['Direction'], axis=1)  # Features
    y = df['Direction']                         # Target variable
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def gradient_boosting_classifier(x_train, x_test, y_train, y_test):
    gbc = GradientBoostingClassifier(loss='log_loss', random_state=42)

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 150],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [2, 3, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
    }

    grid = GridSearchCV(
        gbc,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        # verbose=1
    )

    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Save model
    joblib.dump(best_model, "gradient_boosting_classifier.pkl")
    print("\nModel saved as 'gradient_boosting_classifier.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'Gradient Boosting Classifier': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/Weekly.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    gradient_boosting_classifier(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
