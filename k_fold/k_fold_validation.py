import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import  cross_val_score


digits = load_digits()
print(dir(digits))

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.3)
print(len(x_train))
print(len(x_test))

def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)


lr = LogisticRegression(max_iter=1000)
lr_score =  get_score(lr,x_train,x_test,y_train,y_test)
print(lr_score)

sv = SVC(max_iter=1000)
sv_score = get_score(sv,x_train,x_test,y_train,y_test)
print(sv_score)


rfc = RandomForestClassifier(n_estimators=100)
rfc_score = get_score(rfc,x_train,x_test,y_train,y_test)
print(rfc_score)



kf = KFold(n_splits=3)
print(kf)


for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(f"{train_index}  {test_index}")


skf = StratifiedKFold(n_splits=3)
print(skf)    

score_l = []
score_svm = []
scores_rf = []

for train_index,test_index in kf.split(digits.data):
    x_train,x_test,y_train,y_test = digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    score_l.append(get_score(lr,x_train,x_test,y_train,y_test))
    score_svm.append(get_score(sv,x_train,x_test,y_train,y_test))
    scores_rf.append(get_score(rfc,x_train,x_test,y_train,y_test))


print(score_l)
print(score_svm)
print(scores_rf)


corss_validation_lr =  cross_val_score(lr,digits.data,digits.target)
print(f"Cross Validation of  Logistic Regression {corss_validation_lr}")

corss_validation_svm =  cross_val_score(sv,digits.data,digits.target)
print(f"Cross validation of SVM {corss_validation_svm}")


corss_validation_rfc =  cross_val_score(rfc,digits.data,digits.target)
print(f"cross validation of RandomForestClassification {corss_validation_rfc}")