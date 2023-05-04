# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training linear regression model using
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#training polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualize lin reg results
# plt.scatter(X,y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color = 'blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.savefig('linreg.png')
# plt.show()



#visualize polynomial regression results
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.savefig('polyreg.png')
plt.show()

#visualize polynomial regression results (for higher resolution and smoother curve)
lin_reg.predict([[6.5]])

#Predicting new result with Linear regression

lin_reg.predict([[6.5]])

#Prediction a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))