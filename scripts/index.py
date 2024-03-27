import numpy as np
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd

data_ = pd.read_csv( 'data/data_from_main.csv' )

print(type(data_),
      type(data_.iloc[0,0]),
      type(data_.iloc[5,5]))

x = data_.values[:,0]
y = data_.values[:,7]

data = list(zip(x, y))

inertias = []

for i in range(1,20):
    
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,20), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show() 


kmeans = KMeans(n_clusters=6)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show() 