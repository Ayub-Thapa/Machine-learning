import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
df = pd.read_csv('/home/phenx-07/Documents/Machine learning/training and testing/carprices.csv')
print(df)
# plt.xlabel('Mileage')
# plt.ylabel('Price')
# plt.scatter(df['Mileage'],df['Sell Price($)'])
# plt.show()

x = df[['Mileage','Sell Price($)']]
y = df[['Age(yrs)']]

print(x)
print(y)

x_train,x_test,y_train,y_test  =train_test_split(x,y,test_size=0.2)
print(len(x_train))
print(x_train)
print(len(x_test))


# create a model
model = linear_model.LinearRegression()
# training a model

model.fit(x_train,y_train)
# predicttion
predict =  model.predict(x_test)
print(predict)
print(y_test)

# accuracy
accuraccy =model.score(x_test ,y_test)
print(f"Accuracy :: {accuraccy*100}%")


