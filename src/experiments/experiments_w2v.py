import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot as plt
import random
import xlrd
import numpy as np
from sklearn.preprocessing import normalize
import math
from collections import Counter
from scipy.spatial.distance import cosine
import re
def get_clients():
    data_path = "E:\Data\kaggle"

    # data_prior = pd.read_csv('{0}/order_products__prior.csv'.format(data_path))
    # data_prior = data_prior[:500000]
    # data_order = pd.read_csv('{0}/orders.csv'.format(data_path))
    # data_product = pd.read_csv('{0}/products.csv'.format(data_path))
    #
    # products = pd.read_csv('{0}/products_v2.csv'.format(data_path))
    # data_product = pd.merge(data_product, products, on=['product_name', 'product_name'])
    #
    # data_aisle = pd.read_csv('{0}/aisles.csv'.format(data_path))
    #
    # data_tmp = pd.merge(data_prior, data_order, on=['order_id', 'order_id'])
    # data_tmp = pd.merge(data_tmp, data_product, on=['product_id', 'product_id'])
    # data_tmp = pd.merge(data_tmp, data_aisle, on=['aisle_id', 'aisle_id'])
    # data_tmp = data_tmp.sort_values(by=['aisle_id'])

    # data = pd.crosstab(data_tmp['order_id'], data_tmp['product_name_groc'])
    transformations = pd.read_csv("../../data/transformations.csv", sep=";", encoding="ISO-8859-1")
    orders = pd.read_csv("../../data/kaggle/order_products__prior.csv")
    orders = orders[:5000000]
    products = pd.read_csv("../../data/kaggle/products.csv")
    kaggle_init = pd.merge(products, orders, on=["product_id", "product_id"])
    kaggle_init = kaggle_init.sort_values(by="order_id")
    kaggle_init = kaggle_init[["product_id", "product_name", "order_id"]]
    names = [i.lower() for i in kaggle_init["product_name"].tolist()]
    kaggle_init["product_name"] = names
    res = pd.merge(kaggle_init, transformations, on=["product_name", "product_name"])
    data_tmp = res.drop_duplicates()
    data_tmp["product_name"] = data_tmp["grocery"]
    print(list(data_tmp))
    data = pd.read_csv('../../data/table_prior_5m.csv')

    print("Files for clients are read")

    df1 = data_tmp[['order_id', 'product_name', 'product_id']]
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
    return clients_aisle, clients_aisle_id

def get_products(clients, x_data, y_data, c_data):
    # flag = 0: return list of recommendation
    # flag = 1: return confidence
    # flag = 2: return item
    buy = []
    # x_data = rules['antecedants'].tolist()
    # # x_data = [list(_x) for _x in x_data]
    # y_data = rules['consequents'].tolist()
    # # y_data = [list(_y) for _y in y_data]
    N = len(rules)
    recommendation_full = []
    recommendation = []
    prob = []
    for rule in x_data:

        if np.array_equal(rule, clients):
            add = y_data[x_data.index(rule)]
            if add not in recommendation_full:
                recommendation_full.append(y_data[x_data.index(rule)])

        check = set(rule).issubset(set(clients))
        # check_2 = list(set(y_data[x_data.index(rule)]) & set(clients))
        if (check):
            add = list(set(y_data[x_data.index(rule)]) - set(clients))
            if add != []:
                for rec_ in add:
                    recommendation.append(rec_)
                prob.append(c_data[x_data.index(rule)])

    # col_names = ['item', 'confidence']
    # rec = pd.DataFrame(columns=col_names)
    # for i in range(len(prob)):
    #     rec = rec.append({'item': recommendation[i], 'confidence': prob[i]}, ignore_index=True)
    # rec = rec.sort_values(by=['confidence'], ascending=False)
    # # rec = rec.as_matrix()


    return recommendation, prob




def distCosine(vecA, vecB):
    def dotProduct(vecA, vecB):
        d = np.multiply(vecA, vecB).sum()
        return d

    a = dotProduct(vecA, vecB)
    b = math.sqrt(dotProduct(vecA, vecA))
    c = math.sqrt(dotProduct(vecB, vecB))
    if (b == 0 or c == 0):
        return 0
    else:
        return a / b / c


def get_recommendation_cos(matrix_cos, client, flag='train'):
    recommendation = []
    names = list(matrix_cos)
    for i in client:
        if i == "artif, sweetener" or i == " artif. sweetener":
            i = "artif. sweetener"
        if i not in names:
            print("ERROR:", i)
        else:
            t = matrix_cos[i].tolist()
            if flag == 'train':
                max_vals_idx = np.argpartition(t, -2)[-2:]
            if flag == 'prior':
                max_vals_idx = np.argpartition(t, -3)[-3:]
            # max_vals = [t[j] for j in max_vals_idx]
            for item in max_vals_idx:
                recommendation.append(names[item])
    return recommendation


def get_recommendation(client, recommendations, x_data, y_data, c_data):
    col_names = ['antecedants', 'consequents', 'confidence']
    recommendation_rules = pd.DataFrame(columns=col_names)
    N = len(x_data)
    x_data_new = []
    y_data_new = []
    c_data_new = []
    for r in range(N):
        ch = list(set(client) & set(x_data[r]))
        ch2 = list(set(recommendations) & set(y_data[r]))
        if ch != [] and ch2 != []:
            x_data_new.append(x_data[r])
            y_data_new.append(y_data[r])
            c_data_new.append(c_data[r])

            #recommendation_rules = recommendation_rules.append(
                #{'antecedants': [x_data[r]], 'consequents': [y_data[r]], 'confidence': c_data[r]}, ignore_index=True)
    result_items, result_conf = get_products(client, x_data_new, y_data_new, c_data_new)
    return result_items, result_conf

def parse_rules(rules, type):
    x_data = rules[type]
    x_data_r = []
    for x in x_data:
        x_d = x[10:len(x) - 1]
        x_d = re.sub("[{})(]", "", x_d)
        t_ = ''
        for st in range(len(x_d)):
            if x_d[st] != "'" and x_d[st - 1] != ',':
                t_ += x_d[st]
        x_d = t_.split(',')
        x_data_r.append(x_d)
    return x_data_r


if __name__ == "__main__":
    # init data (grocery or extended grocery). format: table
    # data = pd.read_csv('../../data/table_train.csv')
    data = pd.read_csv('../../data/table_kaggle_adapted.csv')
    data = data.drop(columns=['order_id'])

    # prior data
    data_prior = pd.read_csv('../../data/table_prior_5m.csv')

    # data = data.drop(columns=['Unnamed: 0'])
    # data_prior = data_prior.drop(columns=['order_id'])

    data_matrix = data.as_matrix()
    data_matrix_prior = data_prior.as_matrix()
    print('Files are read')

    names = (list(data))
    names_prior = (list(data_prior))

    N = len(data_matrix[0])
    print(len(data_matrix), len(data_matrix[1]))
    print(names)
    Matrix_cos = [[0 for x in range(N)] for y in range(N)]

    N_prior = len(data_matrix_prior[0])
    print(len(data_matrix_prior), len(data_matrix_prior[1]))
    print(names_prior)

    Matrix_cos_prior = [[0 for x in range(N_prior)] for y in range(N_prior)]

    print("ok2")

    for i in range(N):
        for j in range(i, N):
            i_i = data_matrix[:, i]
            j_j = data_matrix[:, j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos[i][j] = cosine
            Matrix_cos[j][i] = cosine
    real_matrix = pd.DataFrame(columns=names, data=Matrix_cos)
    print('Matrix_cos - ok - data')

    for i in range(N_prior):
        for j in range(i, N_prior):
            i_i = data_matrix_prior[:, i]
            j_j = data_matrix_prior[:, j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos_prior[i][j] = cosine
            Matrix_cos_prior[j][i] = cosine
    prior_matrix = pd.DataFrame(columns=names_prior, data=Matrix_cos_prior)
    print('Matrix_cos - ok - prior')

    clients_aisle, clients_aisle_id = get_clients()
    print(len(clients_aisle))

    # rules = pd.read_csv('../../data/rules_train.csv')
    rules = pd.read_csv('../../data/rules_kaggle_adapted.csv')
    # rules = rules[:100000]
    x_data = parse_rules(rules, 'antecedants')
    y_data = parse_rules(rules, 'consequents')
    c_data = rules['confidence'].tolist()
    print('data rules - ok', len(rules))

    rules_prior = pd.read_csv('../../data/rules_prior_5m.csv')
    # rules_prior = rules[:100000]

    x_data_prior = parse_rules(rules_prior, 'antecedants')
    y_data_prior = parse_rules(rules_prior, 'consequents')
    c_data_prior = rules_prior['confidence'].tolist()

    print('prior rules - ok', len(rules_prior))

    C = 1000#len(clients_aisle)
    pr_array = []
    rec_arr = []
    f1_arr = []
    conf = []
    intersection = []
    n_rec_prior = []
    n_rec_train = []
    for i in range(C):
        print(i + 1, '/', C)
        print("Client", clients_aisle[i])
        clients_aisle_id = []

        cos = get_recommendation_cos(real_matrix, clients_aisle[i])
        cos = list((set(cos) - set(clients_aisle[i])))

        cos_prior = get_recommendation_cos(prior_matrix, clients_aisle[i], flag='prior')
        cos_prior = list((set(cos_prior) - set(clients_aisle[i])))

        print('T clients - {0}; cos - {1}'.format(len(clients_aisle[i]), len(cos)))
        print('P clients - {0}; cos - {1}'.format(len(clients_aisle[i]), len(cos_prior)))

        result_item, result_conf = get_recommendation(clients_aisle[i], cos, x_data, y_data, c_data)
        result_item = list(set(result_item))
        print("item rec ok")
        result_item_prior, result_conf_prior = get_recommendation(clients_aisle[i], cos_prior, x_data_prior,
                                                                  y_data_prior, c_data_prior)
        result_item_prior = list(set(result_item_prior))
        print("prior item rec ok")

        print(set(result_item_prior), set(result_item))

        inter = len(list(set(result_item) & set(result_item_prior)))
        if len(result_item):
            recall = inter/len(result_item)
        else:
            recall = 0
        if len(result_item_prior):
            precision = inter/len(result_item_prior)
        else:
            precision = 0
        if precision+recall:
            f1 = 2*precision*recall/(precision+recall)
        else:
            f1 = 0
        pr_array.append(precision)
        rec_arr.append(recall)
        f1_arr.append(f1)
        n_rec_prior.append(len(result_item_prior))
        n_rec_train.append(len(result_item))
        union = len(list(set(result_item_prior+result_item)))
        u = -1
        if union != 0:
            u = float(inter) / float(union)
            intersection.append(u)
            if len(result_conf) != 0:
                a_conf = sum(result_conf)/len(result_conf)
            else:
                a_conf = 0
            if len(result_conf_prior) != 0:
                a_prior_conf = sum(result_conf_prior) / len(result_conf_prior)
            else:
                a_prior_conf = 0
            a = a_conf-a_prior_conf
            conf.append(a)
            print("Intersection:", u)
            print("Confidence:", a)
        else:
            print("No recommendation")


    # np.savetxt("../../data/rec_all_rules_train.csv", rec_arr, delimiter=";")
    # np.savetxt("../../data/pr_all_rules_train.csv", pr_array, delimiter=";")
    # np.savetxt("../../data/f1_all_rules_train.csv", f1_arr, delimiter=";")
    # np.savetxt("../../data/inter_all_rules_train.csv", intersection, delimiter=";")
    # np.savetxt("../../data/conf_all_rules_train.csv", conf, delimiter=";")
    # np.savetxt("../../data/number_recommendation_all_rules_prior.csv", n_rec_prior, delimiter=";")
    # np.savetxt("../../data/number_recommendation_all_rules_train.csv", n_rec_train, delimiter=";")

    np.savetxt("../../data/rec_all_rules_train_adapt.csv", rec_arr, delimiter=";")
    np.savetxt("../../data/pr_all_rules_train_adapt.csv", pr_array, delimiter=";")
    np.savetxt("../../data/f1_all_rules_train_adapt.csv", f1_arr, delimiter=";")
    np.savetxt("../../data/inter_all_rules_train_adapt.csv", intersection, delimiter=";")
    np.savetxt("../../data/conf_all_rules_train_adapt.csv", conf, delimiter=";")
    # np.savetxt("../../data/number_recommendation_all_rules_prior.csv", n_rec_prior, delimiter=";")
    np.savetxt("../../data/number_recommendation_all_rules_train_adapt.csv", n_rec_train, delimiter=";")

