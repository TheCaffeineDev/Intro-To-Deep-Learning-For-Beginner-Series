import pandas as pd  #it will help to read our dataset

import matplotlib.pyplot as plt   #it will let us visualize our models and data
from sklearn import linear_model   #machine learning library we are using for this example


#read  data

dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data

body_reg=linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

#visualize results

plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()

