import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()
print(dir(digits))

plt.gray()

for i in range(4):
    plt.matshow(digits.images[i])


df = pd.DataFrame(digits.data)    
df['target'] = digits.target
print(df.head())

x_train,x_test,y_train,y_test = train_test_split(df.drop(['target'],axis='columns'),df.target,test_size=0.2)
print(len(x_train))
print(len(x_test))
 
model =  RandomForestClassifier(n_estimators=40)

model.fit(x_train,y_train)
accuracy =  model.score(x_test,y_test)
print(accuracy)
y_pred = model.predict(x_test) 
print(y_pred)
# y_test is truth and y_pred is predicted 
cm = confusion_matrix(y_test,y_pred)
plt.xlabel ('Predicted')
plt.ylabel ('Truth')
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)

plt.show()