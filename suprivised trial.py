import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

'''
We can use Scikit learn to build a simple linear regression model
you can use it use it like 
model = LinearRegression() 
see the documentation here: 
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

To get Mean Squared Error (MSE) you can use 
mean_squared_error(Actual,Prediction)
'''


#### Your Code Goes Here ####

## Step 1: Load Data from CSV File ####

dataframe = pd.read_csv("student_scores(AutoRecovered).csv")
print(dataframe.describe())

## Step 2: Plot the Data ####
x = dataframe["Hours"].values.reshape(-1,1)
y = dataframe["Scores"].values.reshape(-1,1)

# plt.plot(x,y,"o")
# plt.show()

## Step 3: Build a Linear Regression Model ####
x_train, y_train = x[0:20], y[0:20]
x_train, y_train = x[19:], y[19:]

print(x_train, y_train)

model = LinearRegression()

model.fit(x_train,y_train)

## Step 4: Plot the Regression line ####
regression_line = model.predict(x)

plt.plot(x,regression_line)

plt.plot(x_train,y_train,"o")
plt.plot(x_train,y_train,"o")
plt.show()

## Step 5: Make Predictions on Test Data ####

y_prediction = (model.predict(8))

## Step 6: Estimate Error ####

print(mean_squared_error(y_test,y_Prediction))

