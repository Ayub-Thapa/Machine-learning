import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('/home/phenx-07/Documents/Machine learning/naive bayes/titanic (1).csv')


df.drop(['PassengerId','Name','Cabin','Embarked','SibSp','Parch','Ticket'],axis='columns',inplace=True)



print(df.head())

target = df.Survived
# print(target)

inputs = df.drop(['Survived'],axis='columns')
# print(inputs)

dummies = pd.get_dummies(inputs.Sex,dtype=int)
print(dummies.head())

inputs = pd.concat([inputs,dummies],axis='columns')
print(inputs.head())
inputs.drop('Sex',axis='columns',inplace=True)
print(inputs.head())

# checking whether value is null or not with any()
print(inputs.columns[inputs.isna().any()])
print(inputs.Age)

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

print(inputs.head(10))

x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)
print(len(x_train))
print(len(x_test))

model = GaussianNB()
model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
print(accuracy)

Y_pred = model.predict(x_test[:10])
print(Y_pred)
print(y_test[:10])

