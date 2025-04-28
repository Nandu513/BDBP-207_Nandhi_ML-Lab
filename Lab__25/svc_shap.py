from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, roc_auc_score
)
from ISLP import load_data
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
import shap

def split_data(df_path, k=50):
    df1 = load_data(df_path)
    X = df1['data']
    y = df1['labels']

    print("Original shape:", X.shape)

    # Filter out labels with frequency < 2
    label_counts = y['label'].value_counts()
    valid_labels = label_counts[label_counts >= 3].index
    mask = y['label'].isin(valid_labels)
    X = X[mask]
    y = y[mask]

    print("After filtering: ",X.shape)

    # Encode labels for use in feature selection
    y_encoded = LabelEncoder().fit_transform(y['label'])

    # Feature Selection: SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    X_kbest = selector.fit_transform(X, y_encoded)
    print(f"After SelectKBest (top {k} features):", X_kbest.shape)

    sel = VarianceThreshold(threshold=0.20)  # Removes low-variance features
    X_var = sel.fit_transform(X_kbest)
    print("After Variance Thresholding:", X_var.shape)

    rf = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=50)
    rf = rf.fit(X_var, y)
    #rf.feature_importances_
    model = SelectFromModel(rf, prefit=True)
    X_new = model.transform(X_var)
    print("Tree based selection:",X_new.shape)

    # Standardize features
    sc = StandardScaler()
    x_scaled = sc.fit_transform(X_new)

    return train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=99,stratify=y_encoded)


def svc_classifier_pca(x_train, x_test, y_train, y_test):
    pipeline = Pipeline([
        ('pca', PCA()),
        ('svc', SVC(probability=True))
    ])

    param_grid = {
        'pca__n_components': [20, 25, 30],  # adjust as needed
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\n[SVC] Best Parameters Found:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    print("Mean cross-validated score of the best_estimator: ",grid.best_score_)

    # Get index of the best model
    best_index = grid.best_index_
    # Get the std of the best model's CV scores
    std_score = grid.cv_results_['std_test_score'][best_index]
    print(f"Standard Deviation of CV scores for Best Estimator: {std_score:.4f}")

    scores = {
        'SVC + PCA': {
            'Train Accuracy': accuracy_score(y_train, best_model.predict(x_train)),
            'Test Accuracy': accuracy_score(y_test, best_model.predict(x_test)),
        }
    }

    print("\nSCORES SUMMARY:")
    print(pd.DataFrame(scores).T)

    return best_model

    # Optional: Save model
    # joblib.dump(best_model, "svc_pca_classifier.pkl")
    # print("\nModel saved as 'svc_pca_classifier.pkl'")


def explain_svc_pca_model(best_model, x_train_pca, feature_names=None, class_index=0):
    """
    Generates SHAP summary plots for a trained SVC + PCA model.
    Assumes PCA is the first step in the pipeline.

    Parameters:
    - best_model: trained pipeline with PCA and SVC
    - x_train_pca: transformed training data (after PCA)
    - feature_names: list of original feature names (optional)
    - class_index: which class to explain (e.g., 0 for the first class)
    """

    # Extract the trained SVC model from the pipeline
    svc_model = best_model.named_steps['svc']

    # Define a function that returns probabilities, as SHAP requires a callable function
    def model_predict_proba(X):
        return svc_model.predict_proba(X)

    # Use SHAP Explainer (passing the callable function for the model)
    explainer = shap.Explainer(model_predict_proba, x_train_pca)
    shap_values = explainer(x_train_pca)

    # Debugging output
    print("Shape of PCA-transformed data (x_train_pca):", x_train_pca.shape)
    print("Shape of SHAP values:", shap_values.values.shape)

    # Ensure SHAP values align with the PCA components and extract the SHAP values for the chosen class
    shap_values_class = shap_values.values[:, :, class_index]  # Select for the chosen class

    # Create feature names for PCA components if not provided
    if feature_names is None:
        # The feature names should be 'PC1', 'PC2', ..., based on the number of PCA components
        feature_names = [f'PC{i + 1}' for i in range(x_train_pca.shape[1])]

    # SHAP Beeswarm Plot (this will show the importance of each PCA component)
    plt.figure(figsize=(10, 6))  # Adjust size for better readability
    plt.title(f"SHAP Beeswarm Plot (Class {class_index} - PCA Components)")
    shap.summary_plot(shap_values_class, x_train_pca, feature_names=feature_names, plot_type="violin")
    plt.show()  # Explicitly show the Violin plot

    # SHAP Bar Plot (Mean |SHAP|) (this will show mean absolute SHAP values for each component)
    plt.figure(figsize=(10, 6))  # Adjust size for better readability
    plt.title(f"SHAP Bar Plot (Class {class_index} - Mean |SHAP|)")
    shap.summary_plot(shap_values_class, x_train_pca, plot_type="bar", feature_names=feature_names)
    plt.show()  # Explicitly show the Bar plot



def main():
    df = "NCI60"
    # x_train, x_test, y_train, y_test = split_data(df)
    x_train, x_test, y_train, y_test = split_data(df, k=2000)
    best_model = svc_classifier_pca(x_train, x_test, y_train, y_test)
    pca = best_model.named_steps['pca']
    # Transform original training data using the trained PCA
    x_train_pca = pca.transform(x_train)
    print("Shape of PCA-transformed data (x_train_pca):", x_train_pca.shape)
    explain_svc_pca_model(best_model, x_train_pca)


if __name__ == "__main__":
    main()
