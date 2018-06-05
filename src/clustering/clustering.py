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



def get_clients():
    data_path = "E:\Data\kaggle"

    data_prior = pd.read_csv('{0}/order_products__prior.csv'.format(data_path))
    data_prior = data_prior[:1000000]
    data_order = pd.read_csv('{0}/orders.csv'.format(data_path))
    data_product = pd.read_csv('{0}/products.csv'.format(data_path))

    products = pd.read_csv('{0}/products_v2.csv'.format(data_path))
    data_product = pd.merge(data_product, products, on=['product_name', 'product_name'])

    data_aisle = pd.read_csv('{0}/aisles.csv'.format(data_path))

    data_tmp = pd.merge(data_prior, data_order, on=['order_id', 'order_id'])
    data_tmp = pd.merge(data_tmp, data_product, on=['product_id', 'product_id'])
    data_tmp = pd.merge(data_tmp, data_aisle, on=['aisle_id', 'aisle_id'])
    data_tmp = data_tmp.sort_values(by=['aisle_id'])

    data = pd.crosstab(data_tmp['order_id'], data_tmp['product_name_groc'])

    print("Files for clients are read")

    df1 = data_tmp[['order_id', 'product_name_groc', 'product_id']]
    df1 = df1.sort_values(by=['order_id'])
    df = df1.as_matrix()

    N = len(data)
    clients_aisle = []
    clients_aisle_id = []
    tmp_array = [df[0][1]]
    tmp_array_id = [df[0][2]]

    for i in range(1, N):
        if (df[i][0] == df[i - 1][0]):
            tmp_array.append(df[i][1])
            tmp_array_id.append(df[i][2])

        else:
            clients_aisle.append(tmp_array)
            clients_aisle_id.append(tmp_array_id)

            tmp_array = [df[i][1]]
            tmp_array_id = [df[i][2]]

    print('clients - ok')
    return data
    #return clients_aisle, clients_aisle_id, data_lbl, data

# data_path = "D:\Data\\retail\kaggle"
data_path_rules = "E:\Projects\MBA_retail\\tmp"
data = pd.read_csv('{0}/change_train.csv'.format(data_path_rules))
data = data.drop(columns=['order_id'])
N_clusters = 4
clusterer = KMeans(n_clusters=N_clusters).fit(data)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(data)
print(len(c_preds), len(data))

col_names =  data.columns
cluster_1  = pd.DataFrame(columns = col_names)
cluster_2  = pd.DataFrame(columns = col_names)
cluster_3  = pd.DataFrame(columns = col_names)
cluster_4  = pd.DataFrame(columns = col_names)
orders = len(c_preds)
for i in range(orders):
    print(i, '/',orders)
    if (c_preds[i]==0):
        cluster_1 = cluster_1.append(data.iloc[[i]])
    if (c_preds[i] == 1):
        cluster_2 = cluster_2.append(data.iloc[[i]])
    if (c_preds[i] == 2):
        cluster_3 = cluster_3.append(data.iloc[[i]])
    if (c_preds[i] == 3):
        cluster_4 = cluster_4.append(data.iloc[[i]])
print('ok')

cluster_1.to_csv('tmp/merge_cluster_1.csv')
cluster_2.to_csv('tmp/merge_cluster_2.csv')
cluster_3.to_csv('tmp/merge_cluster_3.csv')
cluster_4.to_csv('tmp/merge_cluster_4.csv')

print(len(cluster_1),len(cluster_2),len(cluster_3),len(cluster_4))


clients_matrix = get_clients()
c_preds = clusterer.predict(clients_matrix)
cluster  = []

for i in range(len(c_preds)):
    print(i, '/',len(c_preds))
    cluster.append(int(c_preds[i]))
print('ok')
np.savetxt("tmp/cluster_clients_prior.csv", cluster, delimiter=";")

print(len(cluster))



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
