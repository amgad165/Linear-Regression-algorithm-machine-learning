import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def loss_function(n, y_predicted, y):
    return (1 / n) * sum([val ** 2 for val in (y_predicted - y)])

def feature_normalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis= 0, ddof = 1)  # Standard deviation (can also use range)
    X_norm = (X - mu)/sigma
    return X_norm

def gradient_descent():
    b0 = 6
    b1 = 5
    b2 = 3
    b3 = 1.5
    for itr in range(iterations):
        print(itr)
        error_cost = 0
        cost_bo = 0
        cost_b1 = 0
        cost_b2 = 0
        cost_b3 = 0
        for i in range(len(X_train)):
            y_train_pred = (b0 + b1 * X1[i] + b2 * X2[i] + b3 * X3[i])
            error_cost = error_cost + (y_train[i] - y_train_pred) ** 2
            for j in range(100):
                partial_wrt_b0 = -2 * (y_train[j] - (b0 + b1 * X1[j] + b2 * X2[j] + b3 * X3[j]))
                partial_wrt_b1 = (-2 * X1[j]) * (y_train[j] - (b0 + b1 * X1[j] + b2 * X2[j] + b3 * X3[j]))
                partial_wrt_b2 = (-2 * X2[j]) * (y_train[j] - (b0 + b1 * X1[j] + b2 * X2[j] + b3 * X3[j]))
                partial_wrt_b3 = (-2 * X3[j]) * (y_train[j] - (b0 + b1 * X1[j] + b2 * X2[j] + b3 * X3[j]))

                cost_bo = cost_bo + partial_wrt_b0
                cost_b1 = cost_b1 + partial_wrt_b1
                cost_b2 = cost_b2 + partial_wrt_b2
                cost_b3 = cost_b3 + partial_wrt_b3

            b0 = b0 - cost_bo * lr
            b1 = b1 - cost_b1 * lr
            b2 = b2 - cost_b2 * lr
            b3 = b3 - cost_b3 * lr

        errors.append(error_cost)
    return b0,b1,b2,b3

house_csv = pd.read_csv("USA_Housing.csv")
house_csv = house_csv.drop(["Address"], axis=1)
house_csv=house_csv.head(1000)
X = house_csv[["Avg. Area House Age","Avg. Area Number of Rooms","Avg. Area Income"]]

y = house_csv["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

b0 = 6
b1 = 5
b2 = 3
b3 = 1.5
X1 = X_train["Avg. Area Income"].values
X2 = X_train["Avg. Area House Age"].values
X3 = X_train["Avg. Area Number of Rooms"].values


X1_test = X_test["Avg. Area Income"].values
X2_test = X_test["Avg. Area House Age"].values
X3_test = X_test["Avg. Area Number of Rooms"].values

# X1_test = np.floor(X1_test)
# X2_test = np.floor(X2_test)
# X3_test = np.floor(X3_test)

X1 = feature_normalize(X1)
X2 = feature_normalize(X2)
X3 = feature_normalize(X3)

X1_test = feature_normalize(X1_test)
X2_test = feature_normalize(X2_test)
X3_test = feature_normalize(X3_test)

y_train = y_train.values
y_test = y_test.values

lr = 0.001
iterations = 30
errors = []
error_cost=0





b0,b1,b2,b3=gradient_descent()

def prediction():
    y_pred = []
    for i in range(len(X_test)):
        y_pred_val = b0 + b1 * X1_test[i] + b2 * X2_test[i] + b3 * X3_test[i]
        y_pred.append(y_pred_val)
    y_pred = np.array(y_pred)
    return y_pred


y_test_predictions = prediction()
for i in range(5):
    print("Y_test Predictions :", y_test_predictions[i], "Y_test :", y_test[i],'\n')

print("W0: ", b0, "\nW1: ", b1, "\nW2 :", b2, "\nW3 :", b3)


# Errors over iterations
plt.figure(figsize=(10,5))
plt.plot(np.arange(1,len(errors)+1),errors,color="red",linewidth=5)
plt.show()