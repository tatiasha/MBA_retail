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

    data_prior = pd.read_csv('{0}/order_products__prior.csv'.format(data_path))
    data_prior = data_prior[:500000]
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
    data_lbl = data_tmp[['product_id', 'product_name_groc']]

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
    return clients_aisle, clients_aisle_id, data_lbl

def get_products(clients, rules, flag):
    # flag = 0: return list of recommendation
    # flag = 1: return confidence
    # flag = 2: return item
    buy = []
    x_data = rules['antecedants'].tolist()
    x_data = [list(_x) for _x in x_data]
    y_data = rules['consequents'].tolist()
    y_data = [list(_y) for _y in y_data]
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
        check_2 = list(set(y_data[x_data.index(rule)]) & set(clients))
        if (check):
            add = list(set(y_data[x_data.index(rule)]) - set(clients))
            if add != []:
                recommendation.append(add)
                prob.append(rules['confidence'][x_data.index(rule)])

    col_names = ['item', 'confidence']
    rec = pd.DataFrame(columns=col_names)
    for i in range(len(prob)):
        rec = rec.append({'item': recommendation[i], 'confidence': prob[i]}, ignore_index=True)
    rec = rec.sort_values(by=['confidence'], ascending=False)
    rec = rec.as_matrix()

    selected = []
    selected_2 = 0

    if recommendation_full != []:
        selected = recommendation_full
    else:
        if recommendation != []:  # !!!!MAX
            # probs = np.array(prob)
            # probs = normalize(probs.reshape(1,-1), 'l1')[0]
            # selected_idx = np.random.choice(len(recommendation), 1, False, probs)[0]
            selected = rec[0][0]  # recommendation[selected_idx]
            selected_2 = rec[0][1]  # prob[selected_idx]

            # if random.random() < probs[selected_idx]:
            # print('!!!!!!!%%%!! ',selected)
            # buy.append(selected)
    recommendation = [item for sublist in recommendation for item in sublist]
    recommendation = list(set(recommendation))
    # recommendation = list((set(recommendation) - set(clients)))

    if (flag == 0):
        return recommendation
    if (flag == 1):
        return selected_2
    if (flag == 2):
        return selected


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


def get_recommendation_cos(Matrix_cos, client, df_lbl):
    recommendation = []
    for i in client:
        a = list(Matrix_cos[i])
        m = -1
        for t in a:
            if t > m and t < 1.0:
                m = t
        r = a.index(m)
        # print(i, '->',r)
        if (r != 0):

            #y = df_lbl.set_index(['product_id'])
            #y = y.loc[r]
            #y = y.as_matrix()
            #recommendation.append(y[0][0])
            recommendation.append(df_lbl[r])


            # print(y[0][0])
        # else:
        # recommendation.append('No Recommendation')
        # print('No Recommendation')
    return recommendation


def get_recommendation(client, recommendations, x_data, y_data, c_data):
    col_names = ['antecedants', 'consequents', 'confidence']
    recommendation_rules = pd.DataFrame(columns=col_names)
    # x_data = rules['antecedants'].tolist()
    # x_data = [list(_x) for _x in x_data]
    # y_data = rules['consequents'].tolist()
    # y_data = [list(_y) for _y in y_data]
    # c_data = rules['confidence'].tolist()
    N = len(x_data)
    for r in range(N):
        ch = list(set(client) & set(x_data[r]))
        ch2 = list(set(recommendations) & set(y_data[r]))
        if (ch != [] and ch2 != []):
            recommendation_rules = recommendation_rules.append(
                {'antecedants': x_data[r], 'consequents': y_data[r], 'confidence': c_data[r]}, ignore_index=True)
    # print(recommendation_rules.head())
    result_confidence = get_products(client, recommendation_rules, 0)
    return result_confidence

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


    data_path_rules = "E:\Projects\MBA_retail\\tmp"
    #data_tmp = pd.read_csv('{0}/data_merge.csv'.format(data_path_rules))
    #data_tmp_prior = pd.read_csv('{0}/data_merge_prior.csv'.format(data_path_rules))


    #data = pd.read_csv('{0}/integrated_train_network.csv'.format(data_path_rules))
    data = pd.read_csv('{0}\\datasets\\change_train.csv'.format(data_path_rules))

    data = data.drop(columns=['order_id'])

    data_prior = pd.read_csv('{0}/datasets/change_prior.csv'.format(data_path_rules))
    #data = data.drop(columns=['Unnamed: 0'])
    data_prior = data_prior.drop(columns=['order_id'])

    data_matrix = data.as_matrix()
    data_matrix_prior = data_prior.as_matrix()
    print('Files are read')

    # df1 = data_tmp[['order_id', 'product_name_groc', 'product_id']]
    # df1 = df1.sort_values(by=['order_id'])
    names = (list(data))
    # df = df1.as_matrix()
    #
    # df1_prior = data_tmp_prior[['order_id', 'product_name_groc', 'product_id']]
    # df1_prior = df1_prior.sort_values(by=['order_id'])
    names_prior = (list(data_prior))
    # df_prior = df1_prior.as_matrix()


    N = len(data_matrix[0])
    print(len(data_matrix), len(data_matrix[1]))
    Matrix_cos = [[0 for x in range(N)] for y in range(N)]

    N_prior = len(data_matrix_prior[0])
    print(len(data_matrix_prior), len(data_matrix_prior[1]))
    Matrix_cos_prior = [[0 for x in range(N_prior)] for y in range(N_prior)]




    cols = ["product_name", "N"]
    products_cos = pd.DataFrame(columns=cols)
    products_cos_prior = pd.DataFrame(columns=cols)

    print("ok2")

    for i in range(N):
        products_cos = products_cos.append({'product_name':names[i], 'N':i},  ignore_index=True)
        for j in range(i, N):
            i_i = data_matrix[:, i]
            j_j = data_matrix[:, j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos[i][j] = cosine
            Matrix_cos[j][i] = cosine
    print('Matrix_cos - ok')

    for i in range(N_prior):
        products_cos_prior = products_cos_prior.append({'product_name':names_prior[i], 'N':i},  ignore_index=True)
        for j in range(i, N_prior):
            i_i = data_matrix_prior[:, i]
            j_j = data_matrix_prior[:, j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos_prior[i][j] = cosine
            Matrix_cos_prior[j][i] = cosine
    print('Matrix_cos - ok - _prior')


    clients_aisle, clients_aisle_id , data_lbl = get_clients()
    print(len(clients_aisle))

    rules = pd.read_csv('{0}/rules/rules_change_train.csv'.format(data_path_rules))
    x_data = parse_rules(rules, 'antecedants')
    y_data = parse_rules(rules, 'consequents')
    c_data = rules['confidence'].tolist()
    print('rules - ok', len(rules))

    rules_prior = pd.read_csv('{0}/rules/rules_prior.csv'.format(data_path_rules))
    x_data_prior = parse_rules(rules_prior, 'antecedants')
    y_data_prior = parse_rules(rules_prior, 'consequents')
    c_data_prior = rules_prior['confidence'].tolist()

    print('rules - ok', len(rules_prior))

    #
    C = 5000#len(clients_aisle)

    conf = []


    for i in range(C):
        print(i + 1, '/', C)
        clients_aisle_id = []
        for j in clients_aisle[i]:
            t = products_cos.loc[products_cos['product_name'] == j.lower()]
            t = t.as_matrix()
            t = t[0][1]
            clients_aisle_id.append(t)
        cos = get_recommendation_cos(Matrix_cos, clients_aisle_id, names)
        cos = list((set(cos) - set(clients_aisle[i])))
        cos_prior = get_recommendation_cos(Matrix_cos_prior, clients_aisle_id, names_prior)
        cos_prior = list((set(cos_prior) - set(clients_aisle[i])))

        print('T clients - {0}; cos - {1}'.format(len(clients_aisle[i]),len(cos)))
        print('P clients - {0}; cos - {1}'.format(len(clients_aisle[i]), len(cos_prior)))

        result = get_recommendation(clients_aisle[i], cos, x_data, y_data, c_data)
        result_prior = get_recommendation(clients_aisle[i], cos_prior, x_data_prior, y_data_prior, c_data_prior)
        print(result_prior, result)
        inter = len(list(set(result) & set(result_prior)))
        union = len(list(set(result_prior+result)))
        if union != 0:
            print(float(inter) / float(union))
            conf.append(float(inter) / float(union))
        else:
            print("0")


    conf = np.sort(conf)

    np.savetxt("{0}/train_statistics_update_v2.csv".format(data_path_rules), conf, delimiter=";")
