from sklearn import datasets, linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
from sklearn . preprocessing import OneHotEncoder

data = pd.read_csv('data_C02_emission.csv')
data = data.drop(["Make", 'Vehicle Class', 'Transmission'], axis=1)
input_variables = ['Engine Size (L)',
                   'Fuel Consumption City (L/100km)',
                   'Cylinders',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Fuel Type'
                   ]

output_variable = ['CO2 Emissions (g/km)']
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type'] = X_encoded

X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print(linearModel.coef_)
y_test_pred = linearModel.predict(X_test)

plt.scatter(y_test, y_test_pred, s=4, c='r')
plt.show()

# najveca pogreska i model koji ju ima
abs = abs(y_test - y_test_pred)
max = np.argmax(abs)
print("maksimalna pogreska: ", abs[max])
info = data.at[data.index[max], 'Model']
print("vozilo s najvecom maksimalnom pogreskom: ", info)
