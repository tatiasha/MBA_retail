import pandas as pd
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
import re
from scipy import spatial
from pympler import asizeof

def get_AR(data, s):
    # i = random.randint(0, 100000)
    # t = []
    # for u in range(i):
    #     t.append(u)
    f_i_data = apriori(data, min_support=s, use_colnames=True)
    end_time_fi = time.time()
    print("data to rules ", asizeof.asizeof(f_i_data))
    if not f_i_data.empty:
        rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    return end_time_fi


def dist_cosine(vecA, vecB):
    def dotProduct(vecA, vecB):
        d = np.multiply(vecA, vecB).sum()
        return d

    a = dotProduct(vecA, vecB)
    b = np.math.sqrt(dotProduct(vecA, vecA))
    c = np.math.sqrt(dotProduct(vecB, vecB))
    if (b == 0 or c == 0):
        return 0
    else:
        return a / b / c


def get_cosine_matrix(data):
    data_matrix = data.as_matrix()

    N = len(data_matrix[0])
    Matrix_cos = [[0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(i, N):
            i_i = data_matrix[:, i]
            j_j = data_matrix[:, j]
            cosine = dist_cosine(i_i, j_j)
            Matrix_cos[i][j] = cosine
            Matrix_cos[j][i] = cosine


def draw_analysis_rs():
    data = []
    for i in range(5):
        t = pd.read_csv("..//..//data//time_AR_N_cos_{0}.csv".format(i))
        t1 = pd.read_csv("..//..//data//time_cos_{0}.csv".format(i))
        t["time_cos"] = t1["time_cos"]
        t = t.drop(columns="Unnamed: 0")
        print(t.head())
        data.append(t)



    x = data[0]["n_orders"].values
    cos_value = []
    cos_err_up = []
    cos_err_down = []

    rules_value = []
    rules_err_up = []
    rules_err_down = []

    top_n_value = []
    top_n_err_up = []
    top_n_err_down = []

    for i in range(len(data[0])):
        cos_v = []
        rules_v = []
        top_n_v = []
        for repeat in range(5):
            cos_v.append((data[repeat]["time_cos"].values)[i])
            rules_v.append((data[repeat]["time_rules"].values)[i])
            top_n_v.append((data[repeat]["time_top_N"].values)[i])

        cos_value.append(sum(cos_v)/len(cos_v))
        cos_err_up.append(max(cos_v) - sum(cos_v)/len(cos_v))
        cos_err_down.append(sum(cos_v)/len(cos_v) - min(cos_v))

        rules_value.append(sum(rules_v)/len(rules_v))
        rules_err_up.append(max(rules_v) - sum(rules_v)/len(rules_v))
        rules_err_down.append(sum(rules_v)/len(rules_v) - min(rules_v))

        top_n_value.append(sum(top_n_v)/len(top_n_v))
        top_n_err_up.append(max(top_n_v) - sum(top_n_v)/len(top_n_v))
        top_n_err_down.append(sum(top_n_v)/len(top_n_v) - min(top_n_v))

    plt.errorbar(x, cos_value, yerr=[cos_err_down, cos_err_up])
    plt.xlabel("number of orders")
    plt.ylabel("time")
    plt.title("cosine matrix")
    plt.show()

    plt.errorbar(x, rules_value, yerr=[rules_err_down, rules_err_up])
    plt.xlabel("number of orders")
    plt.ylabel("time")
    plt.title("associative rules")
    plt.show()

    plt.errorbar(x, top_n_value, yerr=[top_n_err_down, top_n_err_up])
    plt.xlabel("number of orders")
    plt.ylabel("time")
    plt.title("frequent items")
    plt.show()


def time_analysis_rs():
    # products = pd.read_csv("..//..//data//kaggle//products.csv")
    # orders = pd.read_csv("..//..//data//kaggle//order_products__train.csv")
    # train = pd.merge(products, orders, on=["product_id", "product_id"])
    data = pd.read_csv("../../data/kaggle_adapted.csv")
    train = pd.crosstab(data['order_id'], data['grocery'])
    print(train.head())
    train = train
    # products_id = train["product_id"].unique()
    products_id = list(train)

    n_products = len(products_id)
    products_step = 10000
    # orders_id = train["order_id"].unique()
    # n_orders = len(orders_id)
    n_orders = len(train)
    orders_step = 10000
    time_top_N = []
    time_rules = []
    time_cos = []

    x_ax = []
    y_ax = []
    # for products in range(products_step, n_products, products_step):
    #     products_list = products_id[:products]
    #     train_cut_products = train.loc[train["product_id"].isin(products_list)]
    #     time_tmp = []
    #     y_ax.append(products)
    for repeat in range(0, 5, 1):
        for order in range(1000, n_orders, orders_step):
            # if order not in x_ax:
            #     x_ax.append(order)
            x_ax.append(order)

            # order_list = orders_id[:order]
            # train_cut_products_orders = train_cut_products.loc[train_cut_products["order_id"].isin(order_list)]

            # train_cut_products_orders = train.loc[train["order_id"].isin(order_list)]
            # data = pd.crosstab(train_cut_products_orders['product_id'], train_cut_products_orders['order_id'])
            data = train[:order]

            print("Calculation AR for {} orders".format(len(data)))
            print("data to FI", asizeof.asizeof(data))
            start_time = time.time()
            end_time_frequent = get_AR(data, 0.005) - start_time #p4
            time_top_N.append(end_time_frequent)
            end_time_rules = time.time() - start_time #p5
            time_rules.append(end_time_rules)

            print("data to cos", asizeof.asizeof(data))
            start_time_cos = time.time()
            get_cosine_matrix(data)  # p7
            end_time_cos = time.time() - start_time_cos
            time_cos.append(end_time_cos)
            print("rules:{0}; fi:{1}; cos:{2}".format(end_time_rules, end_time_frequent, end_time_cos))
            # time_tmp.append(all_time)

            print("Calculation AR for {0} orders is ended".format(len(data)))

            # time_arr.append(time_tmp)

        # data = pd.DataFrame(time_arr, columns=x_ax, index=y_ax)
        # sns.heatmap(data, annot=True, linewidths=.5, cbar=True)
        td = pd.DataFrame()
        td["n_orders"] = x_ax
        # td["time_top_N"] = time_top_N
        # td["time_rules"] = time_rules
        td["time_cos"] = time_cos

        td.to_csv("..//..//data//time_cos_{0}.csv".format(repeat))
        plt.show()


def get_words(external_data, original_data):
    words = []
    tmp = []
    sentences = []
    for product in external_data:
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if (len(word) > 2 and word != 'no') and not word.isdigit():
                if word.find("("):
                    word = word.replace("(", "")
                if word.find(")"):
                    word = word.replace(")", "")
                if word.find("'s"):
                    word = word.replace("'s", "")
                if word.find("s'"):
                    word = word.replace("s'", "")
                if word.find("."):
                    word = word.replace(".", "")
                if word.find("d'"):
                    word = word.replace("d'", "")
                if word.find("\\"):
                    word = word.replace("\\", "")
                if len(word) > 2 and not word.isdigit():
                    words.append(word)
                    tmp.append(word)
        sentences.append(tmp)
        tmp = []
    tmp = []
    for product in original_data:
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if (len(word) > 2 and word != 'no') and not word.isdigit():
                if word.find("("):
                    word = word.replace("(", "")
                if word.find(")"):
                    word = word.replace(")", "")
                if word.find("'s"):
                    word = word.replace("'s", "")
                if word.find("s'"):
                    word = word.replace("s'", "")
                if word.find("."):
                    word = word.replace(".", "")
                if word.find("d'"):
                    word = word.replace("d'", "")
                if word.find("\\"):
                    word = word.replace("\\", "")
                if len(word) > 2 and not word.isdigit():
                    words.append(word)
                    tmp.append(word)
        sentences.append(tmp)
        tmp = []

    return list(set(words)), sentences


def get_data(path, name_to_return, delimiter=',', names=None):
    data = pd.read_csv(path, sep=delimiter, names=names)
    return data[name_to_return]


def process_original_data(data):
    product_name = []
    for product in data:
        product_name.append(product.lower())
    return product_name


def transform_data(data):
    stoplist = ['8\"', "18.2z"]
    snowball_stemmer = SnowballStemmer("english")
    new_data = []
    for product in data:
        new_product = ""
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if not word.isdigit() and word not in stoplist:
                if word.find("("):
                    word = word.replace("(", "")
                if word.find(")"):
                    word = word.replace(")", "")
                if word.find("'s"):
                    word = word.replace("'s", "")
                if word.find("s'"):
                    word = word.replace("s'", "")
                if word.find("."):
                    word = word.replace(".", "")
                if word.find("d'"):
                    word = word.replace("d'", "")
                word = snowball_stemmer.stem(word)
                if len(word) > 2 and not word.isdigit():
                    new_product += word + " "
        if len(new_product[:-1]) > 2:
            new_data.append(new_product[:-1])
        else:
            new_data.append("-1")
    return new_data


def process_external_data(data):
    data = data.tolist()
    product_name = []
    for order in data:
        o = order.split(",")
        for word in o:
            if word not in product_name:
                product_name.append(word)
    return product_name


def get_matrix(data, model):
    matrix = []
    voc = list(model.wv.vocab)
    l = len(model[voc[0]])
    for product in data:
        p = product.split(" ")
        vec = [0 for _ in range(l)]
        product_u = []
        for elem in p:
            if elem in voc and elem not in product_u:
                product_u.append(elem)
                vec += model[elem]
        matrix.append(vec)
        # print(vec)
    return matrix


def preprocessing(n):
    external_path = "../../data/groceries/groceries.csv"
    original_path = "../../data/kaggle/products.csv"
    external_data = get_data(external_path, 'products', delimiter=';', names=['products'])
    n_ext = len(external_data)*n
    external_data = external_data[:int(n_ext)]
    t = asizeof.asizeof(external_data)
    external_data = process_external_data(external_data)
    original_data = get_data(original_path, 'product_name')
    t += asizeof.asizeof(original_data)
    print(t)
    n_orig = len(original_data)*n
    original_data = original_data[:int(n_orig)]
    original_data = process_original_data(original_data.tolist())

    transformed_external_data = transform_data(external_data)
    transformed_original_data = transform_data(original_data)

    all_unique_words, sentences = get_words(transformed_external_data, transformed_original_data)

    return original_data, external_data, sentences


def word2vec_(sentences_):
    if sentences_:
        model = Word2Vec(sentences_, min_count=10, compute_loss=True, iter=1000, size=200)
        return model
    else:
        return -1


def comparing_(model, original_data, external_data):
    original_matrix = get_matrix(original_data, model)
    external_matrix = get_matrix(external_data, model)

    step = 0

    for product_id in range(len(original_matrix)):
        ma = [-100, -100, -100]
        ma_idx = [-100, -100, -100]
        for query_id in range(len(external_matrix)):
            d = original_matrix[product_id]
            diff = 1 - spatial.distance.cosine(d, external_matrix[query_id])
            for m in range(len(ma)):
                if diff > ma[m]:
                    ma[m] = diff
                    ma_idx[m] = query_id
                    break
        step += 1



def draw_analysis_uni():
    data = []
    for i in range(5):
        t = pd.read_csv("..//..//data//time_unification_{0}.csv".format(i))
        # t = t.drop(columns="Unnamed: 0")
        print(t.head())
        data.append(t)

    x = data[0]["Size"].values
    cos_value = []
    cos_err_up = []
    cos_err_down = []

    rules_value = []
    rules_err_up = []
    rules_err_down = []

    top_n_value = []
    top_n_err_up = []
    top_n_err_down = []

    for i in range(len(data[0])):
        cos_v = []
        rules_v = []
        top_n_v = []
        for repeat in range(5):
            cos_v.append((data[repeat]["Pre-processing"].values)[i])
            rules_v.append((data[repeat]["W2V model"].values)[i])
            top_n_v.append((data[repeat]["Comparing items"].values)[i])

        cos_value.append(sum(cos_v)/len(cos_v))
        cos_err_up.append(max(cos_v) - sum(cos_v)/len(cos_v))
        cos_err_down.append(sum(cos_v)/len(cos_v) - min(cos_v))

        rules_value.append(sum(rules_v)/len(rules_v))
        rules_err_up.append(max(rules_v) - sum(rules_v)/len(rules_v))
        rules_err_down.append(sum(rules_v)/len(rules_v) - min(rules_v))

        top_n_value.append(sum(top_n_v)/len(top_n_v))
        top_n_err_up.append(max(top_n_v) - sum(top_n_v)/len(top_n_v))
        top_n_err_down.append(sum(top_n_v)/len(top_n_v) - min(top_n_v))

    plt.errorbar(x, cos_value, yerr=[cos_err_down, cos_err_up])
    plt.xlabel("number of orders")
    plt.ylabel("time")
    plt.title("Pre-processing")
    plt.show()

    plt.errorbar(x, rules_value, yerr=[rules_err_down, rules_err_up])
    plt.xlabel("number of orders")
    plt.ylabel("time")
    plt.title("W2V model")
    plt.show()

    plt.errorbar(x, top_n_value, yerr=[top_n_err_down, top_n_err_up])
    plt.xlabel("number of orders")
    plt.ylabel("time")
    plt.title("Comparing items")
    plt.show()





if __name__ == "__main__":
    time_analysis_rs()
    # draw_analysis_uni()
    # for repeat in range(1):
    #     x = []
    #     time_pre = []
    #     time_model = []
    #     time_comp = []
    #     for i in range(1, 11, 1):
    #         n = i/10.0
    #         start_time_pre = time.time()
    #         original_data, external_data, sentences = preprocessing(n)
    #         print("{0}_Pre-processing: {1}% {2}".format(repeat, n*100, time.time()-start_time_pre))
    #         print(asizeof.asizeof(sentences))
    #         start_time_model = time.time()
    #         model = word2vec_(sentences)
    #         print("{0}_W2V model: {1}% {2}".format(repeat, n*100, time.time()-start_time_model))
    #         start_time_comp = time.time()
    #         print(asizeof.asizeof(model)+asizeof.asizeof(original_data)+asizeof.asizeof(external_data))
    #         if model != -1:
    #             comparing_(model, original_data, external_data)
    #         print("{0}_Comparing items: {1}% {2}".format(repeat, n*100, time.time()-start_time_comp))
    #         end_time = time.time()
    #
    #         x.append(len(original_data)+len(external_data))
    #         print(len(original_data)+len(external_data))
    #         time_pre.append(start_time_model-start_time_pre)
    #         time_model.append(start_time_comp-start_time_model)
    #         time_comp.append(end_time-start_time_comp)
    #
    #     # d = {"Size": x, "Pre-processing": time_pre, "W2V model": time_model, "Comparing items": time_comp}
    #     # time_data = pd.DataFrame(d)
    #     # time_data.to_csv("../../data/time_unification_{}.csv".format(repeat), index=None)









