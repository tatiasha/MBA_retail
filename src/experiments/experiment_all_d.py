import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
from sklearn.cluster import KMeans
import math
import re

data_path = "E:\Projects\MBA_retail\\tmp"
N_clusters = 4


def convert_to_list2(a):
    tmp = a
    tmp = tmp[2:-2]
    tmp2 = tmp.split('\r')
    tmp2 = [item for sublist in tmp2 for item in sublist]
    tmp2 = [x for x in tmp2 if (x != ' ' and x != '\n')]
    tmp2 = list(map(int, tmp2))
    return tmp2


def get_train_data():
    data_path_train = 'E:\Projects\MBA_retail\\tmp'
    baskets_train = pd.read_csv('{0}/table_all_dunnhumby_d2t.csv'.format(data_path_train))

    baskets = baskets_train['BASKET_ID'].tolist()
    cust_prod = baskets_train.drop(columns=['BASKET_ID'])
    return cust_prod, baskets

def get_data():
    data_path2 = "E:\Data\dunnhumby"

    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path2))
    # transactions = pd.read_csv('E:\Projects\MBA_retail\\tmp\\transactions_dunnhumby.csv'.format(data_path))

    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path2))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path2))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    total = total.sort_values(by="BASKET_ID")
    # total = total.drop(columns=['Unnamed: 0'])
    return total

def get_clients():
    data_path_train = 'E:\Projects\MBA_retail\\tmp'
    data_path = "E:\Data\dunnhumby"
    baskets_train = pd.read_csv('{0}/test_input_target_dunnhumby.csv'.format(data_path_train))
    baskets_list = baskets_train['BASKET_ID']

    data = get_data()
    vals = data['PRODUCT_ID'].value_counts().to_frame()
    vals["Fre"] = vals.index
    products = pd.DataFrame()
    products['PRODUCT_ID'] = vals["Fre"]
    products['Frequency'] = vals["PRODUCT_ID"]
    products = products.loc[products['Frequency'] > 500]
    data = pd.merge(products, data, on=['PRODUCT_ID', 'PRODUCT_ID'])

    coss = pd.merge(baskets_train, data, on=['BASKET_ID', 'BASKET_ID'])
    coss = coss.drop_duplicates()
    cust_prod = pd.crosstab(coss['BASKET_ID'], coss['PRODUCT_ID'])

    clients_aisle = []
    clients_aisle_id = []
    tmp = []
    aisle = []
    aisle_id = []
    products_id = data['PRODUCT_ID'].unique().tolist()
    for basket in baskets_list:
        items = data.loc[data["BASKET_ID"] == basket]
        items = items['PRODUCT_ID'].tolist()
        clients_aisle.append(items)
        for i in items:
            tmp.append(products_id.index(i))
            if i not in aisle:
                aisle.append(i)
                aisle_id.append(products_id.index(i))
        clients_aisle_id.append(tmp)
        tmp = []
    data_lbl = pd.DataFrame()
    data_lbl['aisle_id'] = aisle_id
    data_lbl['aisle'] = aisle


    return clients_aisle, clients_aisle_id, data_lbl, cust_prod, baskets_list


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
        # check = set(rule).issubset(set(clients))
        check = list(set(rule) & set(clients))
        if (check != []):
            add = list(set(y_data[x_data.index(rule)]) - set(clients))
            if add != []:
                recommendation.append(add)
                prob.append(rules['confidence'][x_data.index(rule)])

    return prob

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

def get_matrix(data_matrix, names):
    #data_matrix = data_matrix.as_matrix()
    N = len(data_matrix[0])
    Matrix_cos = [[0 for x in range(N)] for y in range(N)]

    cols = ["product_name", "N"]
    products_cos = pd.DataFrame(columns=cols)

    for i in range(N):
        products_cos = products_cos.append({'product_name': names[i], 'N': i}, ignore_index=True)
        for j in range(i, N):
            i_i = data_matrix[:, i]
            j_j = data_matrix[:, j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos[i][j] = cosine
            Matrix_cos[j][i] = cosine
    return Matrix_cos

def get_recommendation_cos(Matrix_cos, client, df_lbl):
    recommendation = []

    for i in client:
        # print(i)
        a = list(Matrix_cos[i])
        m = -1
        for t in a:
            if t > m and t < 1.0:
                m = t
        r = a.index(m)
        # print(i, '->', r)
        #if (r != 0):
            #y = df_lbl.set_index(['aisle_id'])
            #y = y.loc[r]
            #y = y.as_matrix()
        y = df_lbl[df_lbl['aisle_id'] == r]['aisle'].tolist()
        if y != []:
            recommendation.append(y)#[0][0])
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

        ch = list(set(client) & set(list(map(int, x_data[r]))))
        ch2 = list(set(recommendations) & set(list(map(int, y_data[r]))))

        if (ch != [] and ch2 != []):
            recommendation_rules = recommendation_rules.append(
                {'antecedants': list(map(int, x_data[r])), 'consequents': list(map(int, y_data[r])), 'confidence': c_data[r]}, ignore_index=True)
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

def prediction(names_cluster_1,names_cluster_2,names_cluster_3,names_cluster_4, clients):
    pred = []
    for i in clients:
        if i in names_cluster_1:
            pred.append(0)
        if i in names_cluster_2:
            pred.append(1)
        if i in names_cluster_3:
            pred.append(2)
        if i in names_cluster_4:
            pred.append(3)
    return pred



if __name__ == "__main__":
    data_path = "E:\Projects\MBA_retail\\tmp"

    clients_aisle, clients_aisle_id, data_lbl, clients_matrix, baskets_list = get_clients()
    print("clients")
    print('rules_start')

    rules_cluster_1 = pd.read_csv('{0}/dunnhumby_rules_all_dunnhumby_c.csv'.format(data_path))
    if (len(rules_cluster_1) > 500000):
        rules_cluster_1 = rules_cluster_1[:500000]
    x_data_1 = parse_rules(rules_cluster_1, 'antecedants')
    y_data_1 = parse_rules(rules_cluster_1, 'consequents')
    c_data_1 = rules_cluster_1['confidence'].tolist()
    print('rules')

    train_data_cluster_1, basket_cluster_1 = get_train_data()
    names_cluster_1 = list(map(int, (list(train_data_cluster_1))))
    train_data_cluster_1 = train_data_cluster_1.as_matrix()
    print("data")

    print(len(names_cluster_1))
    matrix_cluster_1 = get_matrix(train_data_cluster_1, names_cluster_1)
    print("matrix_cluster")


    number_clients = 21000#len(clients_aisle)
    conf = []
    for c in range(number_clients):
        Matrix_cos = matrix_cluster_1
        x_data = x_data_1
        y_data = y_data_1
        c_data = c_data_1
        names = names_cluster_1

        print('{0}/{1}'.format(c+1, number_clients))
        print("len(client) - ", len(clients_aisle[c]))

        cos = get_recommendation_cos(Matrix_cos, clients_aisle_id[c], data_lbl)
        if len(cos) != 0:
            cos = list(set(sum(cos, [])) - set(clients_aisle[c]))
        else:
            cos = list(set(cos) - set(clients_aisle[c]))
        print('len(cos) - ', len(cos))
        result = get_recommendation(clients_aisle[c], cos, x_data, y_data, c_data)

        if len(result) != 0:
            print(float(sum(result))/float(len(result)))
            conf.append(float(sum(result))/float(len(result)))
        else:
            print(0)
            conf.append(0)

    np.savetxt("confidence_clusters_d_all.csv", conf, delimiter=";")
