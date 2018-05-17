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

# data_path = "D:\Data\\retail\kaggle"
data_path = "E:\Data\kaggle"
prior = pd.read_csv('{0}/order_products__train.csv'.format(data_path))

N = len(prior)

#prior = prior[0:1000]

products = pd.read_csv('{0}/products.csv'.format(data_path))

aisles = pd.read_csv('{0}/aisles.csv'.format(data_path))

orders = pd.read_csv('{0}/orders.csv'.format(data_path))

#print(aisles.head())
#print(prior.head())
#print(products.head())
#print(orders.head())


mt = pd.merge(prior,products, on = ['product_id','product_id'])
mt = pd.merge(mt,aisles, on = ['aisle_id','aisle_id'])
mt = pd.merge(mt,orders, on = ['order_id','order_id'])


mt_sort = mt['aisle'].value_counts()
print(mt_sort.head())
mt_sort = mt_sort.to_frame()

cust_prod = pd.crosstab(mt['order_id'], mt['aisle']) #user_id

N_components = 50
''''
pca = PCA(n_components=N_components)
pca.fit(cust_prod)
pca_samples = pca.transform(cust_prod)

ps = pd.DataFrame(pca_samples)

tocluster = pd.DataFrame(ps)
'''
'''''
for i in range(N_components):
    for j in range(i+1,N_components):
        fig = plt.figure()
        plt.plot(tocluster[i], tocluster[j], 'o', markersize=2, color='blue', alpha=0.5)
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.title("PCA components = "+str(i)+" and "+str(j))
        plt.show()
        fig.savefig('img/order_id_pca_'+str(i)+'_'+str(j)+'.png')
'''

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

cluster_1.to_csv('tmp/train_cluster_1.csv')
cluster_2.to_csv('tmp/train_cluster_2.csv')
cluster_3.to_csv('tmp/train_cluster_3.csv')
cluster_4.to_csv('tmp/train_cluster_4.csv')

print(len(cluster_1),len(cluster_2),len(cluster_3),len(cluster_4))

#colors = ['orange', 'b', 'y', 'k', 'pink', 'yellow', 'green', 'purple', 'cyan', 'grey']
#colored = [colors[k] for k in c_preds]

''''
for i in range(N_components):
    for j in range(i+1,N_components):
        fig = plt.figure()
        plt.scatter(tocluster[i],tocluster[j],  color = colored, s = 10)
        for ci,c in enumerate(centers):
            plt.plot(c[i], c[j], 'x', markersize=8, color='red', alpha=0.9)
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.title("KMeans n_clusters = "+ str(N_clusters)+" components ="+str(i)+" and "+str(j))
        plt.show()
        fig.savefig('img/order_id_kmeans_'+str(i)+'_'+str(j)+'.png')
'''

#clust_prod = cust_prod.copy()
#clust_prod['cluster'] = c_preds


#cl1 = clust_prod[clust_prod['cluster']==3]
#print(cl1)
#cl1 = cl1.sort_values(ascending=False)
#cl1 = cl1.to_frame()
#mt_cl1 = cl1.join(mt_sort)
#mt_cl1.to_csv('experi.csv', sep=';', encoding='utf-8') 
#mt_cl1.columns = ['a', 'b']
#mt_cl1 = mt_cl1.sort_values(by=['b'],ascending=False)




# for i in range(N_clusters):
#     print("cluster #", i)
#     print("Sort by second column")
#     cl1 = clust_prod[clust_prod['cluster']==i].drop('cluster',axis=1).mean()
#     cl1 = cl1.sort_values(ascending=False)
#     cl1.to_csv('tmp/clustering_withoutPCA_'+'_c'+str(i)+'_aisle_sort_order.csv', sep=';', encoding='utf-8')
#     print(cl1.head())
#     cl1 = cl1.to_frame()
#     mt_cl1 = cl1.join(mt_sort)
#     mt_cl1.columns = ['a', 'b']
#     mt_cl1 = mt_cl1.sort_values(by=['b'],ascending=False)
#     #print("Sort by third column")
#     #print(mt_cl1.head())
#     mt_cl1.to_csv('tmp/clustering_withoutPCA_'+'_c'+str(i)+'_aisle_sort_global_order.csv', sep=';', encoding='utf-8')
