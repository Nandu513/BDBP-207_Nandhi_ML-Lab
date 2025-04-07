import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(data):
    df=pd.read_csv(data,header=None)
    # pd.set_option("display.max_columns",None)
    # print(df.head())

    df[60] = df[60].map({'M': 1, 'R': 0}).astype(int)

    # print("\n=== Label Encoding Check ===")
    # print(df.iloc[:, -1].unique())
    # print(df.iloc[:, -1].dtype)
    # print(df.dtypes)
    # print(df.head(2))

    return df

def split_data(data):

    x_train,x_test=train_test_split(data,test_size=0.3,shuffle=True,random_state=42)

    return x_train,x_test

def kfold(data):
    df=load_data(data)
    print(df.head())
    print("Unique labels:", df.iloc[:, -1].unique())
    print("Target dtype:", df.iloc[:, -1].dtype)

    x_train,x_test=split_data(df)

    kf=KFold(n_splits=10,shuffle=True,random_state=42)

    x=x_train.iloc[:,:-1]
    y=x_train.iloc[:,-1]

    sc=StandardScaler()
    x_sc=sc.fit_transform(x)

    best_model=None
    scores=[]

    for i,(train_index,val_index) in enumerate(kf.split(x_sc,y)):
        print(f'Fold: {i}')
        x_train_fold,x_val=x_sc[train_index],x_sc[val_index]
        y_train,y_val=y.iloc[train_index],y.iloc[val_index]

        model=LogisticRegression(penalty='l2', tol=0.000001, C=1.0,
                                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                                 random_state=42, solver='lbfgs', max_iter=1000, multi_class='deprecated',
                                 warm_start=False, n_jobs=None, l1_ratio=None)

        model.fit(x_train_fold,np.ravel(y_train))
        y_pred=model.predict(x_val)

        score = accuracy_score(y_val, y_pred)
        scores.append(score)
        print(f'score: {score}')

        if i == 9:
            best_model = model

    print(f"Mean Accuracy: {np.mean(scores)}")
    print(f"Standard Deviation: {np.std(scores)}")

    y_test = x_test.iloc[:,-1]
    x_test=sc.transform(x_test.iloc[:,:-1])
    y_test_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_score}")


def main():
    df=r"C:/Users/Naga Nandi Reddy/Downloads/sonar.csv"
    kfold(df)

if __name__=="__main__":
    main()
