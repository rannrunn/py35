import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# print pd.DataFrame(diabetes.data).head()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]  # keeps only the 3rd column

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()  # Create linear regression object
regr.fit(diabetes_X_train, diabetes_y_train)  # Train the model using the training sets

diabetes_y_pred_lin = regr.predict(diabetes_X_test)

print('Coefficients:', regr.coef_, regr.intercept_)  # The coefficients
print("Mean square error (MSE): %.2f"
      % np.mean((diabetes_y_pred_lin - diabetes_y_test) ** 2))  # The mean square error
print
'R^2: %.2f' % regr.score(diabetes_X_test, diabetes_y_test)  # Explained variance score

# Plot outputs
plt.clf()
plt.scatter(diabetes_X_train, diabetes_y_train, color='black', alpha=0.1)
plt.scatter(diabetes_X_test, diabetes_y_test, color='red')
plt.scatter(diabetes_X_test, diabetes_y_pred_lin, color='green')  # benchmark
plt.grid(True)
plt.show()
# Predictions against ground truth
plt.scatter(diabetes_y_test, diabetes_y_pred_lin)
plt.plot(diabetes_y_test, diabetes_y_test)
plt.show()