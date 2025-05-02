#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Let's import the dataset
dataset=pd.read_csv(r'E:\emp_sal.csv')

#Lets divide the dataset into independent and dependent variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fiting the KNN model to the dataset
from sklearn.neighbors import KNeighborsRegressor
#model=KNeighborsRegressor()
regressor_knn=KNeighborsRegressor(n_neighbors=4,weights='distance')
regressor_knn.fit(X,y)

y_pred_knn=regressor_knn.predict([[6.5]])
print(y_pred_knn)


# Visualising the KNN Results
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor_knn.predict(X), color = 'blue')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position level')    
plt.ylabel('Salary')
plt.show()



