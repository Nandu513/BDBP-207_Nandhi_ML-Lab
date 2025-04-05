import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def load_data(data):
    df = pd.read_csv(data)
    print(df.head())

    df.drop(columns=['id'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({"B": 0, "M": 1})

    x_train, x_test = train_test_split(df, test_size=0.3, random_state=42)

    return x_train, x_test


def find_best_hyperparameters(x_train, y_train):
    """ Find the best lambda (regularization strength) and eta0 (learning rate) before training. """
    lambda_values = [0.0001, 0.001, 0.01, 0.1]  # Regularization strength
    eta_values = [0.001, 0.01, 0.1]  # Learning rates
    l1_ratio = 0.5  # Fixed balance of L1 & L2 regularization

    best_lambda = None
    best_eta0 = None
    best_score = 0

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for lambda_elastic in lambda_values:
        for eta0 in eta_values:
            scores = []
            for train_idx, val_idx in kf.split(x_train):
                x_train_fold, x_val = x_train[train_idx], x_train[val_idx]
                y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=lambda_elastic,
                                    l1_ratio=l1_ratio, learning_rate='constant', eta0=eta0,
                                    max_iter=1000, random_state=42)
                clf.fit(x_train_fold, y_train_fold)
                y_pred = clf.predict(x_val)
                scores.append(accuracy_score(y_val, y_pred))

            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_lambda = lambda_elastic
                best_eta0 = eta0

    print(f"Best Hyperparameters: lambda = {best_lambda}, eta0 = {best_eta0}")
    return best_lambda, best_eta0


def logistic_regression(data):
    x_train, x_test = load_data(data)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    x = x_train.drop(columns=['diagnosis'], axis=1)
    y = x_train['diagnosis']

    sc = StandardScaler()
    x_tr = sc.fit_transform(x)

    # **Find the best lambda and eta0 once before training**
    best_lambda, best_eta0 = find_best_hyperparameters(x_tr, y)
    print(best_lambda,best_eta0)

    scores = []
    best_clf = None  # Store best model

    for i, (train_index, val_index) in enumerate(kf.split(x_tr, y)):
        print(f"Fold {i}:")
        x_train_fold, x_val = x_tr[train_index], x_tr[val_index]
        y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

        clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=best_lambda,
                            l1_ratio=0.5, learning_rate='constant', eta0=best_eta0,
                            max_iter=1000, random_state=42)
        clf.fit(x_train_fold, y_train_fold)
        y_pred = clf.predict(x_val)

        score = accuracy_score(y_val, y_pred)
        scores.append(score)

        if i == 4:
            best_clf = clf

    print(f"Mean Accuracy: {np.mean(scores)}")
    print(f"Standard Deviation: {np.std(scores)}")

    # **Test Set Evaluation (Moved Outside Loop)**
    x_test_scaled = sc.transform(x_test.drop(columns=['diagnosis'], axis=1))
    y_test = x_test['diagnosis']
    y_test_pred = best_clf.predict(x_test_scaled)
    test_score = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_score}")


def main():
    df = "C:/Users/Naga Nandi Reddy/Downloads/breast-cancer.csv"
    logistic_regression(df)


if __name__ == '__main__':
    main()
