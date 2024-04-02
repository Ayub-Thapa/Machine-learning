import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC

iris = load_iris()
print(dir(iris))

df = pd.DataFrame(iris.data,columns = iris.feature_names)
print(df.head())
df['flower'] = iris.target
df['flower']=df['flower'].apply(lambda x : iris.target_names[x])
print(df.head())


x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)

print(len(x_train))
print(len(x_test))

rs = RandomizedSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20,30],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)

rs.fit(iris.data,iris.target)

# print(rs.cv_results_)
# print(dir(rs))
df = pd.DataFrame(rs.cv_results_)
print(df)
