# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Data
dataset=pd.read_csv(r'E:\Investment.csv')

#Divide into Dependent & Independent variables
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

# Lets fill categorical variables with dummies
x=pd.get_dummies(x,dtype=int)

#Divide Data into test & train data in 80:20 ratio
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#Lets Build Model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Lets predict y_test
y_pred=regressor.predict(x_test)

# Slope
m_coef=regressor.coef_
print(m_coef)

# Constant
c_inter=regressor.intercept_
print(c_inter)

#Lets add constants to x with ones or with constant variables
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) #--> ones

# Lets import API statsmodels ,method leastsquares and call the model OLS 
import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]


#OrdinaryLeastSquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit() #endog=dependent variable ,exog=independent
regressor_OLS.summary()


#Recuresive Feature Elimination(backward elimination)
#Lets Delete the variables having more than 0.05 p value
import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,5]]
#ordinaryleastsquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3]]
#ordinaryleastsquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,3]]
#ordinaryleastsquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1]]
#ordinaryleastsquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


# Investing in DigitalMarketing can give the profits is the prediction of the model

bias=regressor.score(x_train,y_train)
print(bias)

regressor=regressor.score(x_test,y_test)
print(regressor)


