import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Function to evaluate model performance
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
    plt.title("Feature Importances - XGBoost Regressor")
    plt.tight_layout()
    plt.show()

# Function to load dataset
def load_data(df_path):
    df = pd.read_csv(df_path)
    x = df.drop(columns=['disease_score'], axis=1)  # Independent variables
    y = df['disease_score']  # Target variable
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

# Function to train XGBoost Regressor
def xgboost_regressor(x_train, x_test, y_train, y_test):
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150],
        'max_depth': [2, 3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Save model
    joblib.dump(best_model, "xgboost_regressor.pkl")
    print("\nModel saved as 'xgboost_regressor.pkl'")

    evaluate(best_model, x_train, x_test, y_train, y_test)

    scores = {
        'XGBoost Regressor': {
            'Train R2': r2_score(y_train, best_model.predict(x_train)),
            'Test R2': r2_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

# Main function
def main():
    df_path = r"C:/Users/Naga Nandi Reddy/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    x_train, x_test, y_train, y_test = load_data(df_path)
    xgboost_regressor(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
