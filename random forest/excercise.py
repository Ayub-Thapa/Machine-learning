import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

iris = load_iris()
print(dir(iris))

df = pd.DataFrame(iris.data,columns=iris.feature_names)

df['target'] = iris.target
print(df.head())
for i,val in enumerate(iris.target_names):
    print(f"{i} : {val}")

x_train,x_test,y_train,y_test = train_test_split(df.drop(['target'],axis='columns'),df.target,test_size=0.2)
print(len(x_train))
print(len(x_test))
model = RandomForestClassifier(n_estimators=30)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(accuracy)
y_pred = model.predict(x_test)
print(y_pred)
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel ('Predicted')
plt.ylabel ('Truth')
plt.show()
