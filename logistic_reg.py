import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from classification_metrics import ClassificationMetrics


def load_data(path):
    df = pd.read_csv(path)
    print(df.head())
    return df

def plot_confusion(cm, labels=["No Disease", "Disease"], title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def logistic_pipeline(df):
    X = df.drop(columns='output')
    y = df['output']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_

    # Predict probabilities and labels
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    # Vary threshold manually (loop over multiple thresholds)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= t).astype(int)
        print(f"\nThreshold = {t}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Metrics from metric.py
    metrics = ClassificationMetrics
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", metrics.accuracy(y_test.to_numpy(), y_pred))
    print("Precision:", metrics.precision(y_test.to_numpy(), y_pred))
    print("Recall (Sensitivity):", metrics.recall(y_test.to_numpy(), y_pred))
    print("Specificity:", metrics.specificity(y_test.to_numpy(), y_pred))
    print("F1 Score:", metrics.f1_score(y_test.to_numpy(), y_pred))

    auc = metrics.roc_curve_plot(y_test.to_numpy(), y_prob)
    print("AUC:", auc)


def main():
    df = load_data("C:/Users/Naga Nandi Reddy/Downloads/heart.csv")
    logistic_pipeline(df)


if __name__ == "__main__":
    main()
