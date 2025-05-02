#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Let's import the dataset
dataset=pd.read_csv(r'E:\emp_sal.csv')

#Lets divide the dataset into independent and dependent variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor_rf=RandomForestRegressor(n_estimators=27,random_state=0)
regressor_rf.fit(X,y)

y_pred_rf=regressor_rf.predict([[6.5]])
print(y_pred_rf)


# Visualising the Results
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor_rf.predict(X), color = 'blue')
plt.title('Truth or Bluff (Random Forest)')
plt.xlabel('Position level')    
plt.ylabel('Salary')
plt.show()


