import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re
from gensim.models import Word2Vec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from pympler import asizeof
import time


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


def divide_matrix(matrix):
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col]:
                matrix[row][col] = 1 / matrix[row][col]
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


def get_words(data):
    words = []
    tmp = []
    sentences = []
    for product in data:
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


def draw_matrix(matrix, words, k, title):
    u, s, v = SVD(matrix, 3)

    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]

    xs = [i[0] for i in x.tolist()]
    ys = [i[0] for i in y.tolist()]
    zs = [i[0] for i in z.tolist()]

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(u)
    y_kmeans = kmeans.predict(u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c=y_kmeans)

    for x, y, z, n in zip(xs, ys, zs, words):  # plot each point + it's index as text above
        ax.text(x, y, z, '%s' % n, size=8, color='black')

    plt.title(title)
    plt.show()


def k_means(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    return y_kmeans


def clustering_word2vec(k, part):
    start_time_pre = time.time()
    init_data = pd.read_csv('../../data/transformations_k2g.csv', delimiter=";")
    init_data["product_name"] = init_data["grocery"]

    transformation = pd.read_csv("..//..//data//groceries_transformation.csv")

    data = pd.merge(transformation, init_data, on=["product_name", "product_name"])
    data = data.drop(columns=["name_1_w", "product_id", "grocery", "Unnamed: 0_x", "Unnamed: 0_y"])
    data = data.sort_values(by="order_id")
    s_d = int(len(data)*part/10.0)
    data = data[:s_d]
    print("ORDERS", len(data['order_id'].unique()))
    print("data_to_pre", asizeof.asizeof(data))

    product_names = data["product_name"].unique()
    product_names_t = data["transformed"].unique()
    orders = data['order_id'].unique()

    terms, sentences = get_words(data["transformed"].values)
    data["sentences"] = sentences

    vec_size = 200
    model = Word2Vec(sentences, min_count=20, compute_loss=True, iter=50, size=vec_size)
    voc = list(model.wv.vocab)

    matrix_pn = []
    step = 0
    for order in orders:
        tmp_data = data.loc[data['order_id'] == order]
        tmp_items = tmp_data['transformed'].unique()
        tmp_vec = [0 for _ in range(vec_size)]
        step += 1
        for product in tmp_items:
            for item in product.split(" "):
                if item in voc:
                    tmp_vec = [a + b for a, b in zip(tmp_vec, model[item])]
        matrix_pn.append(tmp_vec)
    end_time_to_pre = time.time()- start_time_pre
    print("time pre: ", end_time_to_pre)
    print("data_to_cluster", asizeof.asizeof(matrix_pn))
    start_time_cluster = time.time()
    pred = k_means(matrix_pn, k)
    end_time_cluster = time.time()-start_time_pre
    print("time cluster:", end_time_cluster)
    d = pd.DataFrame()
    d['cluster'] = pred
    d['values'] = matrix_pn
    # draw_matrix(matrix_pn, product_names, k, "w2v")
    print("data_to_int", asizeof.asizeof(d))

    return d


def get_external_w2v():
    init_data = pd.read_csv('../../data/orders_grocery.csv', delimiter=",")
    transformation = pd.read_csv("..//..//data//groceries_transformation.csv")

    data = pd.merge(transformation, init_data, on=["product_name", "product_name"])
    data = data.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y", "Unnamed: 0.1"])
    data = data.sort_values(by="order_id")

    product_names = data["product_name"].unique()
    product_names_t = data["transformed"].unique()
    orders = data['order_id'].unique()

    terms, sentences = get_words(data["transformed"].values)
    data["sentences"] = sentences

    vec_size = 200
    model = Word2Vec(sentences, min_count=20, compute_loss=True, iter=5, size=vec_size)
    voc = list(model.wv.vocab)
    matrix_pn = []
    step = 0
    for order in orders:
        tmp_data = data.loc[data['order_id'] == order]
        tmp_items = tmp_data['transformed'].unique()

        step += 1
        tmp_vec = [0 for _ in range(vec_size)]
        for product in tmp_items:
            for item in product.split(" "):
                if item in voc:
                    tmp_vec = [a + b for a, b in zip(tmp_vec, model[item])]
        matrix_pn.append(tmp_vec)

    return matrix_pn


def w2v_clustering_distance(k, part):
    original = clustering_word2vec(k, part)
    external = get_external_w2v()
    list_clusters = original['cluster'].unique()

    # dst_external = {}
    start_time_int = time.time()
    arr = []
    for d in external:
        min_dst = 100000000
        for c in list_clusters:
            cluster_data = original.loc[original['cluster'] == c]
            products = cluster_data['values'].values
            dst = [euclidean(d, i) for i in products]
            mean_dst = sum(dst) / len(dst)
            if mean_dst < min_dst:
                min_dst = mean_dst
                # cl = c
        # dst_external[d] = (min_dst, cl)
        arr.append(min_dst)
    end_time_int = time.time()-start_time_int
    print("time int:", end_time_int)
    return sum(arr) / len(arr)


if __name__ == "__main__":
    for i in range(1,11, 1):
        t = w2v_clustering_distance(15, i)
    print(t)
