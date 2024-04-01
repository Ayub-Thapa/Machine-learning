import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/home/phenx-07/Documents/Machine learning/k means/income.csv')
print(df.head())

# plt.scatter(df['Age'], df['Income($)'])

# km = KMeans(n_clusters=3)

# y_pred = km.fit_predict(df[['Age', 'Income($)']])
# print(y_pred)

# df['cluster'] = y_pred
# print(df.head())



scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df[['Income($)']] = scaler.transform(df[['Income($)']])

# Scaling 'Age'
scaler.fit(df[['Age']])
df[['Age']] = scaler.transform(df[['Age']])

km = KMeans(n_clusters=3)

y_pred = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = y_pred
print(df)

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='red', label='Cluster 1')
plt.scatter(df2.Age, df2['Income($)'], color='green', label='Cluster 2')
plt.scatter(df3.Age, df3['Income($)'], color='blue', label='Cluster 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='black', marker='*', label='Centroids')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()



sse = []
for i in range (1,10):
    km = KMeans(n_clusters=i)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

print(sse)
plt.plot(range (1,10),sse)

plt.show()