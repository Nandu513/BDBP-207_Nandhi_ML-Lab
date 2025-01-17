import pandas as pd
import numpy as np

def read_data(file_name):
    data=pd.read_csv(file_name)
    print()
    print("Printing 1st few rows: \n" ,data.head())
    print()
    print("Dimensions of given data: \n No of rows:", data.shape[0],"\n No of columns: ", data.shape[1])
    print()
    print("Viewing data type of each column/feature:\n",data.dtypes)
    print()
    print("Info of the features:")
    print(data.info())
    print()
    print("Checking for null values: \n", data.isnull().sum())
    print()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print("Describe of data:\n",data.describe(include="all"))
    print()
    print("Checking for duplicates in data:\n", data.duplicated().sum())
    print()
    return data


def split_data(file_name):
    data = read_data(file_name)
    rand_data=data.sample(frac=1)
    total_rows=data.shape[0]
    ratio=0.75
    train_size=int(ratio*total_rows)
    train_data=rand_data[0:train_size]
    test_data=rand_data[train_size:]
    x_train = train_data.drop(['disease_score', 'disease_score_fluct', 'Gender'], axis=1)
    y_train = train_data["disease_score"]
    x_test= test_data.drop(['disease_score', 'disease_score_fluct', 'Gender'], axis=1)
    y_test= test_data["disease_score"]

    return x_train,x_test,y_train,y_test

def hypothesis_func(file):
    x_train,x_test,y_train,y_test=split_data(file)
    dim=x_train.shape[1]
    x_train.insert(0, "x_0", 1)
    x_test.insert(0, "x_0", 1)
    x_train_matrix=np.asmatrix(x_train[:].values)
    x_test_matrix=np.asmatrix(x_test[:].values)
    y_train_matrix=np.asmatrix(y_train[:].values)
    y_test_matrix=np.asmatrix(y_test[:].values)
    parameters=np.array([[1]]*(dim+1))
    res=np.dot(x_train_matrix,parameters)




def main():
    df="/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    read_data(df)
    split_data(df)
    hypothesis_func(df)


if __name__ == '__main__':
    main()
