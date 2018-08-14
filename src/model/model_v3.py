import pandas as pd
import numpy as np
import math
import re
import random
from sklearn.decomposition import TruncatedSVD

data_path_rules = "E:\Projects\MBA_retail\\tmp\\rules"
data_path = "E:\Data\kaggle"
changed_data_path = "E:\Projects\MBA_retail\\tmp"
cols = ["product_name", "N"]

products_cos = pd.DataFrame(columns=cols)

def get_clients():

    data_prior = pd.read_csv('{0}/order_products__train.csv'.format(data_path))
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

    print("Files for clients are read")

    df1 = data_tmp[['order_id', 'product_name_groc', 'product_id']]
    df1 = df1.sort_values(by=['order_id'])
    df = df1.as_matrix()

    N = len(data)
    clients_aisle = []
    # clients_aisle_id = []
    tmp_array = [df[0][1]]
    tmp_array_id = [df[0][2]]

    for i in range(1, N):
        if (df[i][0] == df[i - 1][0]):
            tmp_array.append(df[i][1])
            tmp_array_id.append(df[i][2])

        else:
            clients_aisle.append(tmp_array)
            # clients_aisle_id.append(tmp_array_id)

            tmp_array = [df[i][1]]
            tmp_array_id = [df[i][2]]

    print('clients - ok')
    return clients_aisle#, clients_aisle_id, data_lbl

def get_products(clients, rules):
    x_data = rules['antecedants'].tolist()
    x_data = [list(_x) for _x in x_data]
    y_data = rules['consequents'].tolist()
    y_data = [list(_y) for _y in y_data]
    recommendation_full = []
    recommendation = []
    prob = []
    for rule in x_data:
        if np.array_equal(rule, clients):
            add = y_data[x_data.index(rule)]
            if add not in recommendation_full:
                recommendation_full.append(y_data[x_data.index(rule)])
        check = set(rule).issubset(set(clients))
        if check:
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

    selected_2 = 0

    if recommendation != []:  # !!!!MAX
        selected_2 = rec[0][1]  # prob[selected_idx]

    recommendation = [item for sublist in recommendation for item in sublist]
    recommendation = list(set(recommendation))

    return recommendation, prob #selected_2

def get_recommendation(client, recommendations, x_data, y_data, c_data):
    col_names = ['antecedants', 'consequents', 'confidence']
    recommendation_rules = pd.DataFrame(columns=col_names)
    N = len(x_data)
    for r in range(N):
        ch = list(set(client) & set(x_data[r]))
        ch2 = list(set(recommendations) & set(y_data[r]))
        if (ch != [] and ch2 != []):
            recommendation_rules = recommendation_rules.append(
                {'antecedants': x_data[r], 'consequents': y_data[r], 'confidence': c_data[r]}, ignore_index=True)
    res, result_confidence = get_products(client, recommendation_rules)
    return res, result_confidence

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
            recommendation.append(df_lbl[r])

    return recommendation

def get_data():
    data_prior = pd.read_csv('{0}/datasets/change_prior.csv'.format(changed_data_path))
    data_prior = data_prior.drop(columns=['order_id'])
    data_matrix_prior = data_prior.as_matrix()
    names_prior = (list(data_prior))
    return data_matrix_prior, names_prior

def distCosine(vecA, vecB):
    def dotProduct(vecA, vecB):
        d = np.multiply(vecA, vecB).sum()
        return d

    a = dotProduct(vecA, vecB)
    b = math.sqrt(dotProduct(vecA, vecA))
    c = math.sqrt(dotProduct(vecB, vecB))
    if b == 0 or c == 0:
        return 0
    else:
        return a / b / c

def get_matrix(data_matrix, names):
    N = len(data_matrix[0])
    print(len(data_matrix), len(data_matrix[1]))
    matrix_cos = [[0 for x in range(N)] for y in range(N)]
    global products_cos
    for i in range(N):
        products_cos = products_cos.append({'product_name':names[i], 'N':i},  ignore_index=True)
        for j in range(i, N):
            i_i = data_matrix[:, i]
            j_j = data_matrix[:, j]
            cosine = distCosine(i_i, j_j)
            matrix_cos[i][j] = cosine
            matrix_cos[j][i] = cosine
    print('Matrix_cos - ok')

    return matrix_cos

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

def get_rules():
    rules_prior = pd.read_csv('{0}\\rules_prior.csv'.format(data_path_rules))
    x_data_prior = parse_rules(rules_prior, 'antecedants')
    y_data_prior = parse_rules(rules_prior, 'consequents')
    c_data_prior = rules_prior['confidence'].tolist()
    return x_data_prior, y_data_prior, c_data_prior


class Client:
    def __init__(self, agent, N):
        self.agent = agent
        self.state = []
        self.reward = -1.0
        self.states = []
        self.state = []
        self.actions = dict()
        self.vector_states = []
        self.transform_states = []
        self.transform_state = []
        if N < 51:
            self.svd = TruncatedSVD(n_components=N, n_iter=7, random_state=51)
            self.flag = True
        else:
            self.flag = False
        self.p = 0


    def get_state(self, prev):
        if (prev >= len(self.states) - 1):
            print("Error. Only {0} clients available".format(len(self.states) - 1))
        else:
            return self.states[prev]

    def get_vector_states(self):
        for i in self.states:
            self.vector_states.append(self.agent.convert_to_vector(i))
        if self.flag:
            self.transform_states = self.svd.fit_transform(self.vector_states)


    def comparing(self, r1, r2):
        inter = 0
        w_0 = 0.05
        w_0_c = 0
        w_1_c = 0
        w_1 = 1
        r = r2
        if self.flag:
            r2 = self.svd.inverse_transform([r2])
            r2 = [round(i) for i in r2[0]]
            for i in range(len(r1)):
                if self.vector_states[self.p][i] == 1:
                    r2[i] = 0

                if r1[i] == r2[i] == 1:
                    inter += w_1
                    w_1_c += 1

                if r1[i] == r2[i] == 0:
                    inter += w_0
                    w_0_c += 1
        else:
            for i in range(len(r1)):
                rnd = random.uniform(0, 1)
                if r2[i] > rnd and self.vector_states[self.p][i] == 0:
                    r2[i] = 1
                else:
                    r2[i] = 0

                if self.vector_states[self.p][i] == 1:
                    r2[i] = 0

                if r1[i] == r2[i] == 1:
                    inter += w_1
                    w_1_c += 1

                if r1[i] == r2[i] == 0:
                    inter += w_0
                    w_0_c += 1

        wx = (len(r1)-w_0_c-w_1_c)/len(r1)
        w_0_c = w_0_c / (len(r1)-sum(r1))
        if sum(r1) == 0 and w_1_c == 0:
            self.reward = 1 - wx
            w_1_c = w_0_c = -1
            return self.reward, r, r1
        if sum(r1) != 0 and w_1_c == 0:
            self.reward = (1 - wx) / sum(r1)
            w_1_c = w_0_c = -1
            return self.reward, r, r1
        if w_0_c == 0:
            self.reward = 1 - wx
            w_0_c = w_1_c = -1
            return self.reward, r, r1
        w_0_c = w_0_c / (len(r1) - sum(r1))
        w_1_c = w_1_c / sum(r1)
        self.reward = 2*w_1_c*w_0_c/(w_1_c+w_0_c)
        return self.reward, r, r1

    def step(self, prev, action_model):
        self.state = self.get_state(prev)
        if self.flag:
            self.transform_state = self.transform_states[prev]
        self.p = prev
        if prev not in self.actions:
            action = self.agent.recommendation(self.state)
            self.actions.update({prev: action})
            ac = action
        else:
            ac = self.actions[prev]

        am = np.array(action_model)
        return self.comparing(ac, am)

        # if sum(ac) == 0:
        #     self.reward = w_0 * inter / len(ac)
        # else:
        #     if w_1_c == 0:
        #         self.reward = w_0*inter
        #     else:
        #         self.reward = inter/(w_1*sum(ac)+w_0*(len(ac)-sum(ac)))
        # precision = float(inter) / len(ac)
        # recall = float(inter)/float(len(am))
        #
        # if precision+recall != 0:
        #     self.reward = 2*precision*recall/(precision+recall)
        # else:
        #     self.reward = 0

        # return self.reward, am, ac

    def reset(self):
        self.state = []
        self.reward = -1.0
        self.states = []
        self.state = []
        self.actions = dict()
        self.agent.reset()
        self.states = get_clients()
        self.vector_states = []
        self.get_vector_states()


class Market:
    def __init__(self):
        self.x_data_model, self.y_data_model, self.c_data_model = [], [], []
        self.matrix_cos_model = []
        self.data, self.names = [], []
        self.clients_aisle_id = []

    def convert_to_vector(self, set_products):
        res = [0 for i in self.names]
        for i in set_products:
            res[self.names.index(i)] = 1
        return res

    def recommendation(self, state):
        self.state = state
        self.f()
        cos_recommendation = get_recommendation_cos(self.matrix_cos_model, self.clients_aisle_id, self.names)
        cos_recommendation = list((set(cos_recommendation) - set(state)))

        model_recommendation, reward = get_recommendation(state, cos_recommendation, self.x_data_model, self.y_data_model, self.c_data_model)
        self.action = model_recommendation #+ self.state

        self.action = self.convert_to_vector(self.action)

        #inter = len(list(set(recommendation) & set(model_recommendation)))
        #precision = float(inter) / float(len(recommendation))
        #recall = float(inter)/float(len(model_recommendation))
        #reward = 2*precision*recall/(precision+recall)

        #self.state = self.convert_to_vector(self.state)

        return self.action

    def f(self):
        global products_cos
        for j in self.state:
            t = products_cos.loc[products_cos['product_name'] == j.lower()]
            t = t.as_matrix()
            t = t[0][1]

            self.clients_aisle_id.append(t)

    def reset(self):
        self.data, self.names = get_data()
        self.matrix_cos_model = get_matrix(self.data, self.names)
        self.x_data_model, self.y_data_model, self.c_data_model = get_rules()
        self.clients_aisle_id = []
        #self.action = []
        self.state = []
