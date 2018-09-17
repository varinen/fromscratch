import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# read the dataset from file
dataset = pd.read_csv('Iris.csv')

print(dataset.head(5))

x = dataset.drop(['Id', 'Species'], axis=1)
x = x.values

wcss = []  # a list to hold the sum of squared distances within clusters

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


print('Optimal number of clusters: 3')

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Claster Sum of Squares')
plt.show()

