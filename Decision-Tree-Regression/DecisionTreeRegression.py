#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Let's import the dataset
dataset=pd.read_csv(r'E:\emp_sal.csv')

#Lets divide the dataset into independent and dependent variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
#model=DecisionTreeRegressor()
regressor_dtr=DecisionTreeRegressor(criterion='absolute_error',splitter='random',random_state=0)
regressor_dtr.fit(X,y)

y_pred_dtr=regressor_dtr.predict([[6.5]])
print(y_pred_dtr)


# Visualising the Results
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor_dtr.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position level')    
plt.ylabel('Salary')
plt.show()

