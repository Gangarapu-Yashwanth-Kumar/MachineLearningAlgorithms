#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Let's import the dataset
dataset=pd.read_csv(r'E:\emp_sal.csv')

#Lets divide the dataset into independent and dependent variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fiting the SVR model to the dataset
from sklearn.svm import SVR
# SVR Model
# Hyper Parameter Tuning
svr_reg=SVR(kernel='poly',degree=4,gamma='auto')
svr_reg.fit(X,y)

#Predicting a new result
svr_pred=svr_reg.predict([[6.5]])
print(svr_pred)


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X,svr_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')    
plt.ylabel('Salary')
plt.show()


