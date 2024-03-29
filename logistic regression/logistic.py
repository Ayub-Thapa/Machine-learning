import pandas as pd
import matplotlib.pyplot as  plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("/home/phenx-07/Documents/Machine learning/logistic regression/insurance_data.csv")
print(df)
plt.scatter(df.age,df.bought_insurance)
# plt.show()
x_train,x_test,y_train,y_test =  train_test_split(df[['age']],df.bought_insurance,test_size=0.2)    

print(x_train)


model = LogisticRegression()
model.fit(x_train,y_train)
test = model.predict(x_test)
print(test)
accuracy = model.score(x_test,y_test)
print(accuracy)
probality = model.predict_proba(x_train)
print(probality)
predictbyAge = model.predict(np.array([[56]]))
print(predictbyAge)