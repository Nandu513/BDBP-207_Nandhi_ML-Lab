import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor


def evaluate(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print("\nTRAINING RESULTS:\n===============================")
    print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"R2 Score: {r2_score(y_train, y_train_pred):.4f}")

    print("\nTESTING RESULTS:\n===============================")
    print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_test_pred):.4f}")

    # Plot true vs predicted
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_test_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Test Set)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.tight_layout()
    plt.show()

    # Feature importance plot
    if hasattr(model, "estimators_") and hasattr(model, "estimators_features_"):
        n_features = x_train.shape[1]
        all_importances = []

        for est, feat_idx in zip(model.estimators_, model.estimators_features_):
            importances = np.zeros(n_features)
            importances[feat_idx] = est.feature_importances_
            all_importances.append(importances)

        avg_importances = np.mean(all_importances, axis=0)

        feat_df = pd.DataFrame({
            'Feature': x_train.columns,
            'Importance': avg_importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=feat_df, x='Importance', y='Feature')
        plt.title("Average Feature Importances (Bagging Regressor)")
        plt.tight_layout()
        plt.show()


def load_data(df_path):
    df = pd.read_csv(df_path)
    x = df.drop(columns=['disease_score'], axis=1)  # Independent variables
    y = df['disease_score']  # Target variable
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def bagging(x_train, x_test, y_train, y_test):
    tree = DecisionTreeRegressor()
    bagging_reg = BaggingRegressor(estimator=tree, random_state=99)

    param_grid = {
        'n_estimators': [50, 100],
        'estimator__max_depth': [3, 5, 7, None],
        'max_samples': [0.8, 1.0],
        'max_features': [0.8, 1.0]
    }

    grid = GridSearchCV(bagging_reg, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Save the best model
    joblib.dump(best_model, "bagging_regressor_model.pkl")
    print("\nModel saved as 'bagging_regressor_model.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'Bagging Regressor': {
            'Train R2': r2_score(y_train, best_model.predict(x_train)),
            'Test R2': r2_score(y_test, best_model.predict(x_test)),
        },
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    bagging(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
