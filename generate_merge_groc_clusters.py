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

if __name__ == "__main__":



    data_kaggle = get_crosstab_kaggle()
    print(len(data_kaggle))
    names = (list(data_kaggle))
    print(len(names))

    data_groc = get_crosstab_groc(names)
    print(len(data_groc))



    N_components = 50

    N_clusters = 4
    clusterer = KMeans(n_clusters=N_clusters).fit(data_groc)
    print("cluterer")
    c_preds = clusterer.predict(data_groc)

    col_names = data_groc.columns
    cluster_1 = pd.DataFrame(columns=col_names)
    cluster_2 = pd.DataFrame(columns=col_names)
    cluster_3 = pd.DataFrame(columns=col_names)
    cluster_4 = pd.DataFrame(columns=col_names)
    orders = len(c_preds)
    for i in range(orders):
        print(i, '/', orders)
        if (c_preds[i] == 0):
            cluster_1 = cluster_1.append(data_groc.iloc[[i]])
        if (c_preds[i] == 1):
            cluster_2 = cluster_2.append(data_groc.iloc[[i]])
        if (c_preds[i] == 2):
            cluster_3 = cluster_3.append(data_groc.iloc[[i]])
        if (c_preds[i] == 3):
            cluster_4 = cluster_4.append(data_groc.iloc[[i]])
    print('ok')

    cluster_1.to_csv('tmp/groceries_cluster_1.csv')
    cluster_2.to_csv('tmp/groceries_cluster_2.csv')
    cluster_3.to_csv('tmp/groceries_cluster_3.csv')
    cluster_4.to_csv('tmp/groceries_cluster_4.csv')

    print(len(cluster_1), len(cluster_2), len(cluster_3), len(cluster_4))