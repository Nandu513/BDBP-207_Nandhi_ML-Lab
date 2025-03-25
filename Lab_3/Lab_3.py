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
    ratio=0.70
    train_size=int(ratio*total_rows)
    train_data=rand_data[0:train_size]
    test_data=rand_data[train_size:]
    x_train = train_data.drop(['disease_score', 'disease_score_fluct', 'Gender'], axis=1)

    # x_train['age']=(x_train['age']-x_train['age'].mean())/x_train['age'].std()
    # x_train['BMI']=(x_train['BMI']-x_train['BMI'].mean())/x_train['BMI'].std()
    # x_train['BP']=(x_train['BP']-x_train['BP'].mean())/x_train['BP'].std()
    # x_train['blood_sugar']=(x_train['blood_sugar']-x_train['blood_sugar'].mean())/x_train['blood_sugar'].std()

    y_train = train_data["disease_score"]
    x_test= test_data.drop(['disease_score', 'disease_score_fluct', 'Gender'], axis=1)
    y_test= test_data["disease_score"]

    x_train.insert(0, "x_0", 1)
    x_test.insert(0, "x_0", 1)
    x_train_matrix = np.array(x_train[:].values)
    x_test_matrix = np.array(x_test[:].values)
    y_train_matrix = np.array(y_train[:].values).reshape(-1, 1)
    y_test_matrix = np.array(y_test[:].values).reshape(-1, 1)

    return x_train_matrix, x_test_matrix, y_train_matrix, y_test_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def linear_regression(x_train,x_test,y_train,y_test):
    scaler=StandardScaler()
    # scalar=scaler.fit(x_train)
    x_train_scalar=scaler.fit_transform(x_train)
    x_test_scalar=scaler.transform(x_test)

    reg_model=LinearRegression()
    reg_model.fit(x_train_scalar,y_train)

    y_pred=reg_model.predict(x_test_scalar)

    r2_value=r2_score(y_test,y_pred)
    print(r2_value)
    print("Coefficients (theta values): \n", reg_model.coef_)

def main():
    df = "/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    x_train,x_test,y_train,y_test=split_data(df)
    linear_regression(x_train,x_test,y_train,y_test)

if __name__ == '__main__':
    main()

