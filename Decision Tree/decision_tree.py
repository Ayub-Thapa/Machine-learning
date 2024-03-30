import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/phenx-07/Documents/Machine learning/Decision Tree/salaries.csv')

print(df.head())

input = df.drop('salary_more_then_100k',axis='columns')
tagert = df['salary_more_then_100k']

print(input)
print(tagert)

lecompany = LabelEncoder()
ledegree = LabelEncoder()
lejob = LabelEncoder()

input['comapny_n'] = lecompany.fit_transform(input['company'])
input['job_n'] = lecompany.fit_transform(input['job'])
input['degree_n'] = lecompany.fit_transform(input['degree'])

print(input.head())

inputs_n = input.drop(['company','job','degree'],axis='columns')
print(inputs_n)

x_train,x_test,y_train ,y_test = train_test_split(inputs_n,tagert,train_size=0.8,)

# models
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,tagert)



accuracy = model.score(inputs_n,tagert)
print(accuracy)

Y_pred = model.predict(np.array([[2,0,0]]))
print(Y_pred)







