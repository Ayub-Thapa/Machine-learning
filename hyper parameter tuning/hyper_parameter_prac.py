import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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

# testing hyper parameter
s_model = SVC(kernel='rbf',C=30,gamma='auto')
s_model.fit(x_train,y_train)
print(s_model.score(x_test,y_test))

# cross val score 

# kernal = ['rbf','linear']
# c = [10,20,30]
# avg_score = {}

# for kval in kernal:
#     for cval in c:
#         cv_score = cross_val_score (SVC(kernel=kval,C=cval,gamma='auto'),iris.data,iris.target,cv=5)
#         avg_score[kval + ' _' +str(cval)] = np.average(cv_score)

# print(avg_score)

clf = GridSearchCV(SVC(gamma='auto'),{
    'C':[10,20,30],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False),

clf.fit(iris.data,iris.target)
# print(clf.cv_results_)

data = pd.DataFrame(clf.cv_results_)
print(data)

print(clf.best_score_)
print(clf.best_params_)

