import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Read the dataset
df = pd.read_csv('/home/phenx-07/Documents/Machine learning/one hot encodding/test5.csv')
print(df)

# One-hot encode the 'town' column
ohe = OneHotEncoder(drop='first')
encoded_town = ohe.fit_transform(df[['town']])

# Concatenate the encoded town with other features
x_ohe = np.concatenate((encoded_town.toarray(), df[['area']].values), axis=1)
print(x_ohe)

# Target variable
yle = df.price

# Train the model
model = linear_model.LinearRegression()
model.fit(x_ohe, yle)

# Predict price for a sample
ok = model.predict(np.array([[0, 1, 3200]]))  # Assuming the first category is 'monroe' and area is 3000
print(ok)
