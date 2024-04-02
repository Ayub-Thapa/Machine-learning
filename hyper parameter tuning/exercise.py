from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

digits = load_digits()
print(dir(digits))
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20, 30],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10, 20],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(multi_class='auto', solver='liblinear'),
        'params': {
            'C': [1, 10, 20, 30]
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(splitter='best'),
        'params': {
            'criterion': ['entropy', 'gini']
        }
    },
    'gaussian': {
        'model': GaussianNB(),
        'params': {
        }
    },
    'multinomialNB': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [1, 5, 10, 20]
        }
    },
}

score = []

for model_names,mp in model_params.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(digits.data,digits.target)
    score.append({
        'model':model_names,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })

# print(score)

df = pd.DataFrame(score)
print(df)