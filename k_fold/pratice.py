from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import  cross_val_score

iris = load_iris()
print(dir(iris))

def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)

corss_validation_lr =  cross_val_score(LogisticRegression(max_iter=1000),iris.data,iris.target)
print(f"Cross Validation of  Logistic Regression {corss_validation_lr}")

corss_validation_svm =  cross_val_score(SVC(max_iter=1000),iris.data,iris.target)
print(f"Cross validation of SVM {corss_validation_svm}")


corss_validation_rfc =  cross_val_score(RandomForestClassifier(n_estimators=40),iris.data,iris.target)
print(f"cross validation of RandomForestClassification {corss_validation_rfc}")


