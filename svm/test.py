import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


digits = load_digits()
print(dir(digits))



for i,val in enumerate(digits.target_names):
    print(f"{i} {val}")


target = digits.target[0:5]


print(target)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model =SVC()
model.fit(x_train , y_train)
accuracy = model.score(x_test,y_test)
print(accuracy)

predict = model.predict(x_train)

print(predict)
