from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
print(dir(iris))

df = pd.DataFrame(iris.data,columns = iris.feature_names)
print(df.head())
df['flower'] = iris.target
df['flower']=df['flower'].apply(lambda x : iris.target_names[x])
print(df.head())


x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)
model_params = {
    'svm':{
        'model':SVC(gamma='auto'),
        'params':{
               'C':[1,10,20,30],
                'kernel':['rbf','linear']
        }
    },
   'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
               'n_estimators':[1,5,10,20],

        }
    },
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear',multi_class='auto'),
        'params':{
                 'C':[1,10,20,30],
        }
    },
}

score = []

for model_names,mp in model_params.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(iris.data,iris.target)
    score.append(
        {
            'model':model_names,
            'best_score': clf.best_score_,
            'best_params' : clf.best_params_
        }
    )

# print(score)  
rdf = pd.DataFrame(score)
print(rdf)  
