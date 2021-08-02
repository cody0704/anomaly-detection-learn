import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # matplotlib 3.2.0 後可省略
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import math

import random

imei = [
    ["35851602","RIM","BlackBerry 8310,smartphone"],
    ["35392201","Motorola","Razr V3i,mobilephone"],
    ["35504700","Nokia","1100","mobilephone"],
    ["35887403","Samsung","SGH-T101G","mobilephone"],
]

def prepare_data(number):
    data = []
    if (number == "35851602"):
        data.append(random.randint(10, 30)) 
        data.append(random.randint(1, 5))
        data.append(1)
    
    if (number == "35392201"):
        data.append(random.randint(10, 30))
        data.append(random.randint(1, 5))
        data.append(2)
    
    if (number == "35504700"):
        data.append(random.randint(120, 150))
        data.append(random.randint(10, 20))
        data.append(3)
    
    if (number == "35887403"):
        data.append(random.randint(300, 500))
        data.append(random.randint(10, 20))
        data.append(4)

    return data

dx = np.empty((0,3), int)
for n in range(1,500):
    dx = np.append(dx, [
        prepare_data(imei[random.randint(0, len(imei)-1)][0])
    ], axis=0)


# prepare data group
clusters = 10
if (len(dx) < 10):
    clusters = len(dates) / 2

# K value range
k_range = range(2, clusters + 1)

distortions = []
scores = []
# get best k value
for i in k_range:
    kmeans = KMeans(n_clusters=i).fit(dx)
    distortions.append(kmeans.inertia_) # 誤差平方和 (SSE)
    scores.append(silhouette_score(dx, kmeans.predict(dx))) # 側影係數

# find k value
selected_K = scores.index(max(scores)) + 2

# re build k means fit
kmeans = KMeans(n_clusters=selected_K).fit(dx)
new_dy = kmeans.predict(dx)

# new dx group
centers = kmeans.cluster_centers_

new_dy

plt.rcParams['font.size'] = 14
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
ax.set_xlabel('Input')
ax.set_ylabel('Start')
ax.set_zlabel('IMEI')
plt.title(f'KMeans={selected_K} groups')
ax.scatter(dx.T[0], dx.T[1], dx.T[2], c=new_dy, alpha=0.5, cmap=plt.cm.Set1)
plt.show()