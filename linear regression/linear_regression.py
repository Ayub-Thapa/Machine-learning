import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
import pickle

# Read data
df = pd.read_csv("/home/phenx-07/Documents/Machine learning/linear regression/csv/test - Sheet1.csv")
d = pd.read_csv("/home/phenx-07/Documents/Machine learning/linear regression/csv/test1 - Sheet1.csv")

# Visualize data
plt.xlabel('area(sqrft)')
plt.ylabel('Price(INR)')
plt.scatter(df.areas, df.price, color='red', marker='+')

# Train linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['areas']], df.price)

# Make prediction for a specific area
prediction = reg.predict(np.array([[3330]]))
print(prediction)

# Print model coefficients
print(reg.coef_)
print(reg.intercept_)

# Print prediction using formula (mx + b)
print(f"mx + b = {reg.coef_ * 3330 + reg.intercept_}")

# Plot regression line
plt.plot(df.areas, reg.predict(df[['areas']]), color='green')
plt.show()

# Make predictions for new data and save to CSV
p = reg.predict(d)
d['price'] = p
d.to_csv('prediction.csv', index=False)

# Save model using pickle
with open('model_pickle', 'wb') as f:
    pickle.dump(reg, f)

# Load model using pickle and make prediction
with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

print(mp.predict(np.array([[4000]])))

# Save model using joblib
joblib.dump(reg, 'model_joblib')

# Load model using joblib and make prediction
mj = joblib.load('model_joblib')

print(mj.predict(np.array([[4000]])))
