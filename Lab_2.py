from sklearn.model_selection import train_test_split
from sklearn   .linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data():
    [x,y]=fetch_california_housing(return_X_y=True)
    return (x,y)

def main():
    [x,y]=load_data()

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=999)

    scalar=StandardScaler()
    scaler=scalar.fit(x_train)
    x_train_scaler=scaler.transform(x_train)
    x_test_scaler=scalar.transform(x_test)

    reg_model=LinearRegression()
    reg_model.fit(x_train,y_train)

    y_pred=reg_model.predict(x_test)

    r2=r2_score(y_test,y_pred)
    print(r2)

if __name__ == '__main__':
    main()

