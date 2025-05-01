#Lets import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Lets read dataset
dataset=pd.read_csv(r'E:\emp_sal.csv')

# DIVIDE THE DATASET INTO INDEPENDENT AND DEPENDENT VARIABLES
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# TRAINING THE MODEL USING LINEAR REGRESSION
from sklearn.linear_model import LinearRegression 
lin_reg= LinearRegression()
lin_reg.fit(X, y)  #training the model

#Visualizing the Linear Regression Graph
plt.scatter(X,y, color= 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title ("Linear Regression Graph")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

lin_model_pred=lin_reg.predict([[6.5]])
lin_model_pred

# TRAINING THE MODEL USING POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
#parameter tuning default degree=2
# poly_reg=PolynomialFeatures()
#hyper-parameter tuning lets increase degree which increases the accuracy of model
# poly_reg=PolynomialFeatures(degree=3)
# poly_reg=PolynomialFeatures(degree=4)
# poly_reg=PolynomialFeatures(degree=5)
poly_reg= PolynomialFeatures(degree=6)
X_poly=poly_reg.fit_transform(X) #transforming the data into polynomial features

poly_reg.fit(X_poly,y)
lin_reg_2= LinearRegression()
lin_reg_2.fit(X_poly,y)  #training the regression model

#Visualizing the Truth or Bluff (Polynomial Regression)
plt.scatter(X,y, color= 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title ("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred