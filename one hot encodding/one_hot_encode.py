import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('/home/phenx-07/Documents/Machine learning/one hot encodding/test5.csv')
print(df)

dummies = pd.get_dummies(df['town'],dtype=int)
print(dummies)

# Concatenate the original DataFrame with the dummy variables along the columns
merged = pd.concat([df, dummies], axis='columns')
print(merged)

final = merged.drop(['town','west windsor'],axis='columns')
print(final)
x = final.drop('price',axis='columns')
print(x)
y =final.price
print(y)
model = linear_model.LinearRegression()
model.fit(x,y)

# predict in  robinsville
robin = model.predict(np.array([[2800,0,1]]))

monroe =  model.predict(np.array([[3000,1,0]]))
weswin =  model.predict(np.array([[3300,0,0]]))

# accuracy
model.score(x,y)

print(robin)
print(monroe)
print(weswin)
print(f"Accuracy {model.score(x,y)*100} % ")


# # one hot encoding
# le = LabelEncoder()
# dfle = df
# dfle.town = le.fit_transform(dfle['town'])
# print(dfle)
# xle = dfle[['town','area']].values #.values returns a 2d array
# print(xle)
# yle = dfle.price
# print(f"===>{yle}")

# ohe = OneHotEncoder(drop='first')
# encoded_town = ohe.fit_transform(df[['town']])

# # Select the relevant column based on the number of categories in 'town'
# x_ohe = encoded_town[:, 1:] 

# model.fit(x_ohe, yle)
# ok = model.predict(np.array([[ 3000 , 0,0]]))
# print(ok)

