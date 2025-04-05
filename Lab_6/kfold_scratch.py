import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_data(file):
    original_df=pd.read_csv(file)
    df=original_df.copy()
    print(df.head())
    print(df.shape)
    diagnosis_map = {'B': 0, 'M': 1}
    df['diagnosis'] = df['diagnosis'].map(diagnosis_map)

    return df

def split_data(file):
    data=load_data(file)
    no_folds=int(input("Enter no of folds you want to split the data into: "))
    rand_data=data.sample(frac=1,random_state=99)
    each_fold=rand_data.shape[0]//no_folds
    split_copy=rand_data.copy()
    i=0
    fold=[]

    while i<no_folds:
        if i==no_folds-1:
            fold.append(split_copy)
            break
        temp_fold=split_copy.iloc[:each_fold,:]
        split_copy.drop(index=split_copy.index[:each_fold],axis=0,inplace=True)
        fold.append(temp_fold)

        i+=1

    # for i in fold:
    #     print(i.shape)
    # print(len(fold))

    return fold,no_folds

def model_for_k_fold(file):
    fold,no_folds=split_data(file)
    scores=[]
    for i in range(no_folds):
        x_test=fold[i].drop(['id','diagnosis'],axis=1)
        y_test=fold[i]['diagnosis']

        train_folds=[j for j in fold if not j.equals(fold[i])]
        train=pd.concat(train_folds)
        # print(train.shape)

        x_train=train.drop(['id','diagnosis'],axis=1)
        y_train=train['diagnosis']

        sc = StandardScaler()
        x_train_scaler = sc.fit_transform(x_train)
        x_test_scaler = sc.transform(x_test)

        lr_model = LogisticRegression()
        lr_model.fit(x_train_scaler, y_train)

        y_pred_lr = lr_model.predict(x_test_scaler)
        score = accuracy_score(y_test, y_pred_lr)
        scores.append(score)
        # print('Accuracy score: {}'.format(score))

    print(scores)
    return scores

def main():
    df="/home/ibab/Downloads/data.csv"
    scores=model_for_k_fold(df)
    print("MEAN_SCORE: ", np.mean(scores))
    print("STD: ",np.std(scores))

if __name__ == "__main__" :
    main()
