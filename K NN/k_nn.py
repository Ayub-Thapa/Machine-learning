import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

iris = load_iris()
print(dir(iris))
print(iris.feature_names)
print(iris.target_names)

df = pd.DataFrame(iris.data,columns=iris.feature_names)

df['target'] = iris.target
print(df.head())

# df1 = df[df['target']==0]
# df2 = df[df['target']==1]
# df3 = df[df['target']==2]
# or
df1 = df[0:50]
df2 = df[50:100]
df3 = df[100:]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='green',marker='+')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red',marker='*')

# plt.show()


x = df.drop(['target'],axis='columns')

x_train,x_test,y_train,y_test = train_test_split(x,df.target,test_size=0.2)
print(len(x_train))
print(len(x_test))

# create knn 
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train,y_train)

print(neigh.score(x_test,y_test))
y_pred = neigh.predict(x_test)
print(y_pred)

cm = confusion_matrix(y_test,y_pred)

print(cm)
print(
classification_report(y_test,y_pred))


