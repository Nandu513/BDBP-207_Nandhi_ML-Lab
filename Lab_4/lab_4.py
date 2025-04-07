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

    x_train['age']=(x_train['age']-x_train['age'].mean())/x_train['age'].std()
    x_train['BMI']=(x_train['BMI']-x_train['BMI'].mean())/x_train['BMI'].std()
    x_train['BP']=(x_train['BP']-x_train['BP'].mean())/x_train['BP'].std()
    x_train['blood_sugar']=(x_train['blood_sugar']-x_train['blood_sugar'].mean())/x_train['blood_sugar'].std()

    y_train = train_data["disease_score"]
    x_test= test_data.drop(['disease_score', 'disease_score_fluct', 'Gender'], axis=1)
    y_test= test_data["disease_score"]

    x_train.insert(0, "x_0", 1)
    x_test.insert(0, "x_0", 1)
    x_train_matrix = np.array(x_train[:].values)
    x_test_matrix = np.array(x_test[:].values)
    y_train_matrix = np.array(y_train[:].values).reshape(-1, 1)
    y_test_matrix = np.array(y_test[:].values).reshape(-1, 1)

    # x_train_matrix = np.asmatrix(x_train[:].values)
    # x_test_matrix = np.asmatrix(x_test[:].values)
    # y_train_matrix = np.array(y_train[:].values).reshape(-1,1)
    # y_test_matrix = np.array(y_test[:].values).reshape(-1,1)
    # print(x_train_matrix.shape)
    # print(x_test_matrix.shape)
    # print(y_train_matrix.shape)
    # print(y_test_matrix.shape)

    return x_train_matrix, x_test_matrix, y_train_matrix, y_test_matrix

def hypothesis_func(x_train,theta):
    y_pred=np.dot(x_train,theta)
    # print(y_pred.shape)
    return y_pred

def cost_func(x_train,y_train,theta):
    y_pred=hypothesis_func(x_train,theta)
    # print(y_pred.shape)
    error=y_pred-y_train
    # print(error.shape)
    sq_error=np.square(error)
    # print(sq_error.shape)
    cost = np.sum(sq_error) / (2 * len(y_train))
    # print(cost)
    return cost

def gradient_descent(x_train, y_train, theta, alpha, iterations, delta=1e-6):
    m = len(y_train)
    cost_values = []
    itr = 0

    for i in range(iterations):
        y_pred = hypothesis_func(x_train, theta)
        # print(y_pred.shape)
        error = y_pred - y_train
        # print(error.shape)
        # print(x_train.T.shape)
        gradient = (1 / m) * np.dot(x_train.T, error)
        # print(gradient.shape)
        theta = theta - alpha * gradient
        # print(theta.shape)
        cost = cost_func(x_train, y_train, theta)
        cost_values.append(cost)

        if i > 0 and abs(cost_values[-1] - cost_values[-2]) < delta:
            y_train_mean = np.mean(y_train,axis=0)
            # print(y_train_mean.shape)
            sst = np.sum((y_train - y_train_mean) ** 2)
            ssr = np.sum((y_pred - y_train) ** 2)
            r2_sq = 1 - (ssr / sst)
            print("Converged at iteration:", i)
            print("initial_cost: ",cost_values[0])
            print("converged cost: ", cost_values[i])
            print("R^2 score:", r2_sq)
            break

    return theta,cost_values

def new_param(file):
    x_train,x_test,y_train,y_test=split_data(file)
    theta = np.zeros((x_train.shape[1], 1))
    alpha=0.1
    iterations=10000
    optimal_theta, cost_values=gradient_descent(x_train,y_train,theta,alpha,iterations)

    return optimal_theta,cost_values

def main():
    df="/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    # read_data(df)
    optimal_theta,cost_values=new_param(df)
    print(optimal_theta)

if __name__ == '__main__':
    main()
