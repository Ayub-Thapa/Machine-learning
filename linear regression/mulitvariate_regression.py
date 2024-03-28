import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('/home/phenx-07/Documents/Machine learning/linear regression/csv/test4.csv')
print(df)
median_val = math.floor(df.bedroom.median())
print(median_val)
df.bedroom = df.bedroom.fillna(median_val)
print(df)

# train model
reg = linear_model.LinearRegression()
reg.fit(df[['areas','bedroom','age']],df.price)
print(reg.coef_)
print(reg.intercept_)

predict =  reg.predict([[2500,3,5]])
print(predict)
prediction = (107.01342282* 2500) + (-6085.5704698 * 3) + (-5169.46308725 * 5) + reg.intercept_
print(prediction)

