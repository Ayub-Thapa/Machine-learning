import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn import linear_model

# Read the original data
df = pd.read_csv('/home/phenx-07/Documents/Machine learning/linear regression/csv/canada_per_capita_income.csv')

# Read the data for which predictions are to be made
d = pd.read_csv('/home/phenx-07/Documents/Machine learning/linear regression/csv/test3.csv')

# Visualize original data
plt.xlabel('year')
plt.ylabel('Per Capita Income')
plt.scatter(df.year, df.per_capita_income, color='red', marker='+')

# Fit the linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df.per_capita_income)

# Make predictions for the new data
predicted_values = reg.predict(d[['year']])
print(predicted_values)

year_to_highlight = 2020
highlight_color = 'black'

# Plot the regression line
plt.scatter(year_to_highlight, reg.predict([[year_to_highlight]]), color=highlight_color, marker='o', label='2020')
plt.plot(df.year,reg.predict(df[['year']]),color='green')

plt.show()
# Save predicted values to a new DataFrame and export to CSV
predicted_df = pd.DataFrame({'year': d['year'], 'predicted_per_capita_income': predicted_values})
predicted_df.to_csv('predicted_per_capita_income.csv', index=False)
