import pandas as pd
import numpy as np
import re
from nltk.stem import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from scipy import spatial
import operator
from matplotlib import pyplot


def get_data(path, name_to_return, delimiter=',', names=None):
    data = pd.read_csv(path, sep=delimiter, names=names)
    return data[name_to_return]


def process_external_data(data):
    data = data.tolist()
    product_name = []
    for order in data:
        o = order.split(",")
        for word in o:
            if word not in product_name:
                product_name.append(word)
    return product_name


def process_original_data(data):
    product_name = []
    for product in data:
        product_name.append(product.lower())
    return product_name


def get_words(external_data, original_data):
    words = []
    for product in original_data:
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if (len(word) > 2 and word != 'no') and word not in words and not word.isdigit():
                words.append(word)

    for product in external_data:
        q = re.split(r'(-|/|,|%|!| |"."|&|™|®)', product)
        for word in q:
            if (len(word) > 2 and word != 'no') and word not in words and not word.isdigit():
                words.append(word)
    return words


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


def get_external_matrix(data, terms):
    A = np.zeros((len(terms), len(data)))
    for line in range(len(data)):
        words = data[line].split(" ")
        for word in words:
            if len(word):
                if word != '-1':
                    idx = terms.index(word)
                    A[idx][line] = 1
    return A


def get_query_matrix(product, terms):
    q = np.zeros((len(terms), 1))
    words = product.split(" ")
    for word in words:
        if len(word):
            if word != '-1':
                idx = terms.index(word)
                q[idx][0] = 1
    return q


def divide_matrix(matrix):
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col]:
                matrix[row][col] = 1/matrix[row][col]
    return matrix


def SVD(A, rank):
    svd = TruncatedSVD(n_components=rank)
    U = svd.fit_transform(A)
    U = np.asmatrix(U)

    S = svd.explained_variance_ratio_
    S = np.diag(S)

    VT = svd.components_
    VT = np.asmatrix(VT)
    V = VT.transpose()  # set of d
    S_1 = divide_matrix(S)
    return U, S_1, V


def LSI(query, U, S_1, V):
    qT = np.asmatrix(query).transpose()
    q_ = np.matmul(qT, U)
    q = np.matmul(q_, S_1)
    q = q.tolist()[0]
    # sim = dict()

    ma_idx = -10
    ma = -100000
    for idx in range(len(V)):
        d = V[idx]
        d = d.tolist()[0]
        diff = 1 - spatial.distance.cosine(d, q)
        # sim.update({idx: diff})
        if diff > ma:
            ma = diff
            ma_idx = idx

    #ma = max(sim.items(), key=operator.itemgetter(1))[0]
    return ma, ma_idx


# 8\" isdigit 18.2z '
if __name__ == "__main__":
    external_path = "../../data/groceries/groceries.csv"
    original_path = "../../data/kaggle/products.csv"

    external_data = get_data(external_path, 'products', delimiter=';', names=['products'])
    external_data = process_external_data(external_data)

    original_data = get_data(original_path, 'product_name')
    original_data = process_original_data(original_data.tolist())
    print("All data are collected. {0} - original products; {1} - external products"
          .format(len(original_data), len(external_data)))

    transformed_external_data = transform_data(external_data)
    transformed_original_data = transform_data(original_data)
    print("All data are transformed. {0} - original products; {1} - external products"
          .format(len(transformed_original_data), len(transformed_external_data)))

    all_unique_words = get_words(transformed_external_data, transformed_original_data)
    print("Unique words are collected - {0} words".format(len(all_unique_words)))
    terms = all_unique_words

    A = get_external_matrix(transformed_external_data, terms)
    # A = get_external_matrix(transformed_original_data, terms)

    print("matrix A")
    U, S_1, V = SVD(A, 168)
    print("SVD(A)")
    x = U[:,0]
    y = U[:,1]
    print(type(U))
    words = terms

    x_ = []
    y_ = []
    w_ = []
    for i in range(len(x)):
        y_.append(y[i].tolist()[0][0])
        x_.append(x[i].tolist()[0][0])
        w_.append(words[i])
    x = x_
    y = y_
    words = w_

    # for line in U[:1000]:
    #     l = line[0].tolist()
    #     x.append(l[0][0])
    #     y.append(l[0][1])
    # print("XY")
    pyplot.scatter(x, y)

    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(x[i], y[i]))
    pyplot.show()

    # matching = dict()
    # kaggle = []
    # grocery = []
    # step = 0
    # closes = []
    # res_orig = []
    # res_external = []
    # for product in transformed_original_data:
    #     query = get_query_matrix(product, terms)
    #     close, close_idx = LSI(query, U, S_1, V)
    #     # print("!!!", close_idx)
    #     step += 1
    #
    #
    #     # matching.update({product: transformed_external_data[close_idx]})
    #     if close_idx == -10 and close == -100000:
    #         kaggle.append(product)
    #         grocery.append("-1")
    #         closes.append("-1")
    #         res_orig.append("-1")
    #         res_external.append(original_data[transformed_original_data.index(product)])
    #     else:
    #         print("{0}/{1}; {2}; {3} = {4}".format(step, len(transformed_original_data), close, product,
    #                                                transformed_external_data[close_idx]))
    #         kaggle.append(product)
    #         grocery.append(transformed_external_data[close_idx])
    #         closes.append(close)
    #         res_orig.append(external_data[close_idx])
    #         res_external.append(original_data[transformed_original_data.index(product)])
    #
    #
    # p = pd.DataFrame()
    # p["kaggle_compressed"] = kaggle
    # p["grocery_compressed"] = grocery
    # p["close_LSI"] = closes
    # p["orig_grocery"] = res_orig
    # p["orig_kaggle"] = res_external
    #
    # p = p.sort_values(by="grocery_compressed")
    # p.to_csv("../../data/transformation_lsi.csv")
    # # print(matching)
    # print("Transformations are ready.{0} products were removed. {1} - before; {2} - after.".format(len(transformed_original_data) - len(p), len(transformed_original_data), len(p)))
