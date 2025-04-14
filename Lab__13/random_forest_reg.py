import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


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

    # Actual vs predicted scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_test_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Test Set)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
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
    plt.title("Feature Importances - Random Forest")
    plt.tight_layout()
    plt.show()

    feature_counts = Counter()
    for tree in model.estimators_:
        feature_indices = tree.tree_.feature
        used = feature_indices[feature_indices >= 0]
        feature_counts.update(used)

    print("\nFeature Usage Count in All Trees:")
    for idx, count in feature_counts.items():
        print(f"Feature '{x_train.columns[idx]}' was used in {count} splits")


def load_data(df_path):
    df = pd.read_csv(df_path)
    x = df.drop(columns=['disease_score'], axis=1)  # Independent variables
    y = df['disease_score']  # Target variable
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def random_forest(x_train, x_test, y_train, y_test):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5, 7, 8, 9, None],
        'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    for i, tree in enumerate(best_model.estimators_[:5]):
        print(f"Tree {i + 1} used max_features: {tree.max_features_}")

    # Save model
    joblib.dump(best_model, "random_forest_regressor.pkl")
    print("\nModel saved as 'random_forest_regressor.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'Random Forest Regressor': {
            'Train R2': r2_score(y_train, best_model.predict(x_train)),
            'Test R2': r2_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)


def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    random_forest(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
