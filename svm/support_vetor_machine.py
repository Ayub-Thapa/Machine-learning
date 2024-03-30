import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
iris =load_iris()
print(dir(iris))

df= pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
for i,val in enumerate(iris.target_names):
    print(f"{i} : {val}")

print(iris.target_names)
# check the target info  according to index which represnt the target_names 
print(df[df.target == 1].head())

df['flower_names'] = df.target.apply(lambda x : iris.target_names[x])
print(df.head())

df0 = df[df.target == 0]
print(df0.head())
df1= df[df.target == 1]
print(df1.head())

df2= df[df.target == 2]
print(df2.head())

plt.xlabel('sepal length (cm')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker= '*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker= '*')
# plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red',marker= '*')
# plt.show()

x = df.drop(['target','flower_names'],axis='columns')
print(x)
y = df.target
print(y)
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = SVC()
model.fit(x_train,y_train)
acurracy = model.score(x_test,y_test)
print(acurracy)