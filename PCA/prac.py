import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.decomposition import PCA

digits = load_digits()
print(dir(digits))


print(digits.data[0])
print(digits.data[0].reshape(8,8))

plt.gray()

plt.matshow(digits.data[9].reshape(8,8))
# plt.show()

print(digits.target[9])
print(np.unique(digits.target))

df = pd.DataFrame(digits.data,columns=digits.feature_names)
print(df.head())
print(df.describe())

x = df
y = digits.target

scaler = StandardScaler()
x_scaled =  scaler.fit_transform(x)
print(x_scaled)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print(score)


# PCA Principal Components Analysis
# pca = PCA(n_components=2) feature are only 2
pca = PCA(0.95) #means 95% calulate and check

x_pca = pca.fit_transform(x)
print(x.shape)

print(x_pca.shape)

print(x_pca)

x_train_pca,x_test_pca,y_train_pca,y_test_pca = train_test_split(x_pca,y,test_size=0.2)

model_pca = LogisticRegression()
model_pca.fit(x_test_pca,y_test_pca)
print(model_pca.score(x_test_pca,y_test_pca))