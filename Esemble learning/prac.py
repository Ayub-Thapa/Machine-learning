import pandas as pd
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from  sklearn.ensemble import BaggingClassifier
from  sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('/home/phenx-07/Documents/Machine learning/Esemble learning/diabetes.csv')

print(df.isnull().sum())

print(df.describe())
print(df.Outcome.value_counts())
x = df.drop('Outcome',axis='columns')
y = df.Outcome
 
print(y)
print(y.value_counts)
print(154/614)

scaler =StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

x_train,x_test,y_train,y_test =train_test_split(x_scaled,y,stratify=y,test_size=0.2)

print(len(x_train))
print(len(x_test))

print(y_train.value_counts())
print(214/400)

model = DecisionTreeClassifier()
cross_score = cross_val_score(model,x,y,cv=5)
print(cross_score.mean())

bag_model = BaggingClassifier(
    # base_estimator=DecisionTreeClassifier(), #deprecated
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=10
)

bag_model.fit(x_train, y_train)
print("OOB Score:", bag_model.oob_score_)
print("score ",bag_model.score(x_test,y_test))

# cross validation of baggingclassifier
score_bag =cross_val_score(bag_model,x,y,cv =5)
print(score_bag.mean())

score_random =cross_val_score(RandomForestClassifier(),x,y,cv =5)
print(score_random.mean())