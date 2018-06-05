import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
from sklearn.cluster import KMeans
import math
import re

data_path = "E:\Projects\MBA_retail\\tmp"
N_clusters = 4

def get_train_data():
    data_path_train = 'E:\Data\kaggle'
    prior = pd.read_csv('{0}/order_products__train.csv'.format(data_path_train))
    products = pd.read_csv('{0}/products.csv'.format(data_path_train))
    aisles = pd.read_csv('{0}/aisles.csv'.format(data_path_train))
    orders = pd.read_csv('{0}/orders.csv'.format(data_path_train))
    mt = pd.merge(prior, products, on=['product_id', 'product_id'])
    mt = pd.merge(mt, aisles, on=['aisle_id', 'aisle_id'])
    mt = pd.merge(mt, orders, on=['order_id', 'order_id'])

    mt_sort = mt['aisle'].value_counts()
    print(mt_sort.head())

    cust_prod = pd.crosstab(mt['order_id'], mt['aisle'])  # user_id
    return cust_prod

def get_clients():
    data_path_train = 'E:\Data\kaggle'
    prior = pd.read_csv('{0}/order_products__prior.csv'.format(data_path_train))
    prior = prior[:50000]
    products = pd.read_csv('{0}/products.csv'.format(data_path_train))
    aisles = pd.read_csv('{0}/aisles.csv'.format(data_path_train))
    orders = pd.read_csv('{0}/orders.csv'.format(data_path_train))
    mt = pd.merge(prior, products, on=['product_id', 'product_id'])
    mt = pd.merge(mt, aisles, on=['aisle_id', 'aisle_id'])
    mt = pd.merge(mt, orders, on=['order_id', 'order_id'])

    mt_sort = mt['aisle'].value_counts()

    cust_prod = pd.crosstab(mt['order_id'], mt['aisle'])  # user_id

    data_lbl = mt[['aisle_id', 'aisle']]

    df1 = mt[['order_id', 'aisle', 'aisle_id']]
    df1 = df1.sort_values(by=['order_id'])
    df = df1.as_matrix()

    N = len(cust_prod)
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
    return clients_aisle, clients_aisle_id, data_lbl, cust_prod

def matrix_cosine(file_path):

    data = pd.read_csv(file_path)
    data_matrix = data.as_matrix()

    N = len(data_matrix[0])

    Matrix_cos = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(i, N):
            i_i = data_matrix[:, i]
            j_j = data_matrix[:, j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos[i][j] = cosine
            Matrix_cos[j][i] = cosine
    return Matrix_cos

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
            y = df_lbl.set_index(['aisle_id'])
            y = y.loc[r]
            y = y.as_matrix()
            recommendation.append(y[0][0])
            # print(y[0][0])
        # else:
        # recommendation.append('No Recommendation')
        # print('No Recommendation')
    return recommendation


def get_recommendation(client, recommendations,  x_data, y_data, c_dat):
    col_names = ['antecedants', 'consequents', 'confidence']
    recommendation_rules = pd.DataFrame(columns=col_names)
    '''''''''''
    x_data = rules['antecedants'].tolist()
    x_data = [list(_x) for _x in x_data]

    y_data = rules['consequents'].tolist()
    y_data = [list(_y) for _y in y_data]
    c_data = rules['confidence'].tolist()
    '''
    N = len(x_data)
    for r in range(N):
        ch = list(set(client) & set(x_data[r]))
        ch2 = list(set(recommendations) & set(y_data[r]))
        if (ch != [] and ch2 != []):
            recommendation_rules = recommendation_rules.append(
                {'antecedants': x_data[r], 'consequents': y_data[r], 'confidence': c_data[r]}, ignore_index=True)
    # print(recommendation_rules.head())
    result_confidence = get_products(client, recommendation_rules, 1)
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
    data_path = "E:\Projects\MBA_retail\\tmp"

    clients_aisle, clients_aisle_id , data_lbl, clients_matrix = get_clients()

    print('rules_start')
    rules_cluster_1 = pd.read_csv('{0}/rules_cluster_1.csv'.format(data_path))
    if (len(rules_cluster_1)>500000):
        rules_cluster_1 =  rules_cluster_1[:500000]
    x_data_1 = parse_rules(rules_cluster_1,'antecedants')
    y_data_1 = parse_rules(rules_cluster_1,'consequents')
    c_data_1 = rules_cluster_1['confidence'].tolist()
    print('cluser 1 - rules')

    rules_cluster_2 = pd.read_csv('{0}/rules_cluster_2.csv'.format(data_path))
    if (len(rules_cluster_2)>500000):
        rules_cluster_2 =  rules_cluster_2[:500000]
    x_data_2 = parse_rules(rules_cluster_2, 'antecedants')
    y_data_2 = parse_rules(rules_cluster_2, 'consequents')
    c_data_2 = rules_cluster_2['confidence'].tolist()
    print('cluster 2 - rules')

    rules_cluster_3 = pd.read_csv('{0}/rules_cluster_3.csv'.format(data_path))
    if (len(rules_cluster_3)>500000):
        rules_cluster_3 =  rules_cluster_3[:500000]
    x_data_3 = parse_rules(rules_cluster_3, 'antecedants')
    y_data_3 = parse_rules(rules_cluster_3, 'consequents')
    c_data_3 = rules_cluster_3['confidence'].tolist()
    print('cluster 3 - rules')

    rules_cluster_4 = pd.read_csv('{0}/rules_cluster_4.csv'.format(data_path))
    if (len(rules_cluster_4)>500000):
        rules_cluster_4 =  rules_cluster_4[:500000]
    x_data_4 = parse_rules(rules_cluster_4, 'antecedants')
    y_data_4 = parse_rules(rules_cluster_4, 'consequents')
    c_data_4 = rules_cluster_4['confidence'].tolist()
    print('cluster 4 - rules')

    print('rules')

    matrix_cluster_1 = matrix_cosine('{0}/train_cluster_1.csv'.format(data_path))
    matrix_cluster_2 = matrix_cosine('{0}/train_cluster_2.csv'.format(data_path))
    matrix_cluster_3 = matrix_cosine('{0}/train_cluster_3.csv'.format(data_path))
    matrix_cluster_4 = matrix_cosine('{0}/train_cluster_4.csv'.format(data_path))
    print('matrix')
    train_data = get_train_data()
    clusterer = KMeans(n_clusters=N_clusters).fit(train_data)

    number_clients = 300#len(clients_aisle)
    conf = []
    c_preds = clusterer.predict(clients_matrix)
    print('prediction')
    for c in range(number_clients):
        if (c_preds[c] == 0):
            Matrix_cos = matrix_cluster_1
            x_data = x_data_1
            y_data = y_data_1
            c_data = c_data_1
        if (c_preds[c] == 1):
            Matrix_cos = matrix_cluster_2
            rules = rules_cluster_2
            x_data = x_data_2
            y_data = y_data_2
            c_data = c_data_2
        if (c_preds[c] == 2):
            Matrix_cos = matrix_cluster_3
            x_data = x_data_3
            y_data = y_data_3
            c_data = c_data_3
        if (c_preds[c] == 3):
            Matrix_cos = matrix_cluster_4
            x_data = x_data_4
            y_data = y_data_4
            c_data = c_data_4

        print('{0}/{1} - cluster{2} - rules = {3}'.format(c+1, number_clients, c_preds[c], len(x_data)))

        cos = get_recommendation_cos(Matrix_cos, clients_aisle_id[c], data_lbl)
        cos = list((set(cos) - set(clients_aisle[c])))
        print('len(cos)', len(cos))
        result = get_recommendation(clients_aisle[c], cos, x_data, y_data, c_data)
        print(result)
        conf.append(result)

    np.savetxt("tmp/confidence_clusters.csv", conf, delimiter=";")