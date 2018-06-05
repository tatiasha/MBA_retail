import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from subprocess import check_output
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
import csv
import random
from sklearn.preprocessing import normalize

from scipy.spatial import distance



data_path = "E:\Data\groceries"

import numpy as np
import csv
import pandas as pd
from sklearn.cluster import KMeans



def readcsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0
    a = []

    for row in reader:
        a.append(row)
        rownum += 1

    ifile.close()
    return a

def rename(init):
    rename_file = readcsv('{0}/rename.csv'.format(data_path))
    init_list = []
    ren_list = []
    rename_file.pop(0)
    for i in rename_file:
        init_list.append(i[0])
        ren_list.append(i[1])

    d = {'init': init_list, 'second': ren_list}
    data_tmp = pd.DataFrame(data=d)

    aisles = []
    for i in init:
        t = data_tmp.loc[data_tmp['init'] == i]
        t = t['second']
        t = t.tolist()
        aisles += t
    return aisles

def get_crosstab_groc(names):
    data_gros = readcsv(data_path + "/groceries.csv")
    data_groc = []
    for i in data_gros:
        x_d = i[0].split(',')
        data_groc.append(x_d)

    indx = 1
    order_id = []
    aisles = []
    not_in = []
    in_in = []
    for i in data_groc:
        for j in i:
            if (j.lower() in names):
                order_id.append(indx)
                aisles.append(j)
                if(j not in in_in):
                    in_in.append(j)
            else:
                if(j not in not_in):
                    not_in.append(j)
        indx += 1


    d = {'order_id': order_id, 'aisle': aisles}
    data_tmp = pd.DataFrame(data=d)
    data = pd.crosstab(data_tmp['order_id'], data_tmp['aisle'])
    return data

def get_crosstab_kaggle():
    data_path_rules = "E:\Projects\MBA_retail\\tmp"
    data = pd.read_csv('{0}/change_train.csv'.format(data_path_rules))
    data = data.drop(columns=['order_id'])
    return data


def transform(data, data_groc):
    print("Transformation")
    a = data.axes
    a = a[1]

    b = data_groc.axes
    b = b[1]

    f = list(data.sum())
    rows = len(f)
    s = sum(f)
    print(s)
    f = f / np.linalg.norm(f)
    print(f)
    #for i in range(len(f)):
        #f[i] = float(f[i])/float(s)

    data_matrix = data.as_matrix().astype(np.float32)
    data_matrix_g = data_groc.as_matrix().astype(np.float32)


    for row in range(len(data_groc)):
        print('groc',row,'/',len(data_groc))
        data_matrix_g[row] = np.float32(data_matrix_g[row] * f)


    for row in range(len(data)):
        print('kaggle',row,'/',len(data))
        data_matrix[row] = np.float32(data_matrix[row] * f)

    col = data.columns
    data = pd.DataFrame(data_matrix, columns=col)
    data_groc = pd.DataFrame(data_matrix_g, columns=col)
    #print(data_groc.head())
    data.to_csv('tmp/transform_train_weight.csv')
    data_groc.to_csv('tmp/transform_groc_weight.csv')

    return data, data_groc

def distance_vec(a,b):
    return distance.euclidean(a,b)

def average_distace_cluster(cluster, idx):
    d = 0
    N = len(idx)
    for i in range(N):
        for j in range(N):
            if i!=j:
                d+=distance_vec(cluster[i], cluster[j])
    d /= N*(N-1)
    return d

def distance_vector_cluster(cluster, idx, vec):
    average = 0
    N = len(idx)
    for i in range(N):
        average += distance_vec(cluster[i], vec)
    average /= N
    return average


if __name__ == "__main__":


    data_kaggle = get_crosstab_kaggle()
    print(len(data_kaggle))
    names = (list(data_kaggle))
    print(len(names))

    data_groc = get_crosstab_groc(names)
    data_groc.to_csv('tmp/changed_groc.csv')
    print(len(data_groc))



    data_kaggle, data_groc = transform(data_kaggle, data_groc)

    N_clusters = 4
    clusterer = KMeans(n_clusters=N_clusters).fit(data_kaggle)
    print("cluterer")
    c_preds = clusterer.predict(data_kaggle)

    col_names = data_kaggle.columns
    cluster_1 = pd.DataFrame(columns=col_names)
    cluster_2 = pd.DataFrame(columns=col_names)
    cluster_3 = pd.DataFrame(columns=col_names)
    cluster_4 = pd.DataFrame(columns=col_names)
    orders = len(c_preds)
    for i in range(orders):
        print(i, '/', orders)
        if (c_preds[i] == 0):
            cluster_1 = cluster_1.append(data_kaggle.iloc[[i]])
        if (c_preds[i] == 1):
            cluster_2 = cluster_2.append(data_kaggle.iloc[[i]])
        if (c_preds[i] == 2):
            cluster_3 = cluster_3.append(data_kaggle.iloc[[i]])
        if (c_preds[i] == 3):
            cluster_4 = cluster_4.append(data_kaggle.iloc[[i]])
    cluster_1 = cluster_1.as_matrix()
    cluster_2 = cluster_2.as_matrix()
    cluster_3 = cluster_3.as_matrix()
    cluster_4 = cluster_4.as_matrix()
    print('Size of clusters')
    print(len(cluster_1), len(cluster_2), len(cluster_3), len(cluster_4))

    idx_cluster_1 = random.sample(range(1, len(cluster_1)), int(0.1 * len(cluster_1)))
    idx_cluster_2 = random.sample(range(1, len(cluster_2)), int(0.1 * len(cluster_2)))
    idx_cluster_3 = random.sample(range(1, len(cluster_3)), int(0.1 * len(cluster_3)))
    idx_cluster_4 = random.sample(range(1, len(cluster_4)), int(0.1 * len(cluster_4)))
    average_cluster_1 = average_distace_cluster(cluster_1, idx_cluster_1)
    average_cluster_2 = average_distace_cluster(cluster_2, idx_cluster_2)
    average_cluster_3 = average_distace_cluster(cluster_3, idx_cluster_3)
    average_cluster_4 = average_distace_cluster(cluster_4, idx_cluster_4)
    print('Average distance in clusters')
    print(average_cluster_1, average_cluster_2, average_cluster_3, average_cluster_4)


    c_preds_groc = clusterer.predict(data_groc)
    data_groc_matrix = data_groc.as_matrix()
    before = len(data_kaggle)
    orders_groc = len(c_preds_groc)
    s = 1.3
    for i in range(orders_groc):
        print("{0}/{1}".format(i, orders_groc))
        if (c_preds_groc[i] == 0):
            d = distance_vector_cluster(cluster_1,idx_cluster_1, data_groc_matrix[i])
            print(d, average_cluster_1)
            if d <= average_cluster_1*s:
                data_kaggle = data_kaggle.append(data_groc.iloc[i])

        if (c_preds_groc[i] == 1):
            d = distance_vector_cluster(cluster_2,idx_cluster_2, data_groc_matrix[i])
            print(d, average_cluster_2)
            if d <= average_cluster_2*s:
                data_kaggle = data_kaggle.append(data_groc.iloc[i])

        if (c_preds_groc[i] == 2):
            d = distance_vector_cluster(cluster_3,idx_cluster_3, data_groc_matrix[i])
            print(d, average_cluster_3)
            if d <= average_cluster_3*s:
                data_kaggle = data_kaggle.append(data_groc.iloc[i])

        if (c_preds_groc[i] == 3):
            d = distance_vector_cluster(cluster_4,idx_cluster_4, data_groc_matrix[i])
            print(d, average_cluster_4)
            if d <= average_cluster_4*s:
                data_kaggle = data_kaggle.append(data_groc.iloc[i])

    print('Size of clusters')
    print(len(cluster_1), len(cluster_2), len(cluster_3), len(cluster_4))
    print('Average distance in clusters')
    print(average_cluster_1, average_cluster_2, average_cluster_3, average_cluster_4)
    print("Before", before)
    print('After',len(data_kaggle))
    print("% of groc", 100.0*(len(data_kaggle)-before)/float(len(data_groc)))
    data_kaggle.to_csv('tmp/extended_change_train_weight.csv')

    f_i_data = apriori(data_kaggle, min_support=0.00001, use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    print('fi ok')
    f_items = f_i_data['itemsets'].tolist()
    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    rules.to_csv('tmp/extended_change_train_rules_weight.csv')

    print('rules - ', len(rules))
