import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from subprocess import check_output
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import csv

data_path = "E:\Projects\MBA_retail\\tmp"
data = pd.read_csv('{0}/change_train.csv'.format(data_path))
data = data.drop(columns=['order_id'])

cust_prod = data

N_components = 50

N_clusters = 4
clusterer = KMeans(n_clusters=N_clusters).fit(cust_prod)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(cust_prod)
print(len(c_preds), len(cust_prod))

col_names =  cust_prod.columns
cluster_1  = pd.DataFrame(columns = col_names)
cluster_2  = pd.DataFrame(columns = col_names)
cluster_3  = pd.DataFrame(columns = col_names)
cluster_4  = pd.DataFrame(columns = col_names)
orders = len(c_preds)
for i in range(orders):
    print(i, '/',orders)
    if (c_preds[i]==0):
        cluster_1 = cluster_1.append(cust_prod.iloc[[i]])
    if (c_preds[i] == 1):
        cluster_2 = cluster_2.append(cust_prod.iloc[[i]])
    if (c_preds[i] == 2):
        cluster_3 = cluster_3.append(cust_prod.iloc[[i]])
    if (c_preds[i] == 3):
        cluster_4 = cluster_4.append(cust_prod.iloc[[i]])
print('ok')

cluster_1.to_csv('tmp/merge_cluster_1.csv')
cluster_2.to_csv('tmp/merge_cluster_2.csv')
cluster_3.to_csv('tmp/merge_cluster_3.csv')
cluster_4.to_csv('tmp/merge_cluster_4.csv')

print(len(cluster_1),len(cluster_2),len(cluster_3),len(cluster_4))

