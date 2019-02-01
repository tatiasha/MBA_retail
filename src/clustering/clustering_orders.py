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
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

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


def clustering_cosine_matrix(k):
    data = pd.read_csv('../../data/table_train.csv')
    data = data.drop(columns=['order_id', 'artif, sweetener'])

    names = (list(data))
    data_matrix = data.as_matrix()

    u, s, v = SVD(data_matrix, 30)
    u = u.tolist()

    #
    # N = len(data_matrix[0])
    # Matrix_cos = [[0 for _ in range(N)] for _ in range(N)]
    #
    # for i in range(N):
    #     for j in range(i, N):
    #         i_i = data_matrix[:, i]
    #         j_j = data_matrix[:, j]
    #         cosine = dist_cosine(i_i, j_j)
    #         Matrix_cos[i][j] = cosine
    #         Matrix_cos[j][i] = cosine

    # draw_matrix(Matrix_cos, names, k, 'cos')

    pred = k_means(u, k)
    d = pd.DataFrame()
    d['cluster'] = pred
    d['values'] = u
    return d


def get_cosine_matrix(path):
    data = pd.read_csv(path)
    data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    table = pd.crosstab(data["order_id"], data["product_name"])
    data_matrix = table.as_matrix()
    u, s, v = SVD(data_matrix, 30)
    u = u.tolist()

    # N = len(data_matrix[0])
    # Matrix_cos = [[0 for _ in range(N)] for _ in range(N)]
    #
    # for i in range(N):
    #     for j in range(i, N):
    #         i_i = data_matrix[:, i]
    #         j_j = data_matrix[:, j]
    #         cosine = dist_cosine(i_i, j_j)
    #         Matrix_cos[i][j] = cosine
    #         Matrix_cos[j][i] = cosine

    return u


def cosine_clustering_distance(k):
    original, s = clustering_cosine_matrix(k)
    external = get_cosine_matrix("..//..//data//orders_grocery.csv")

    list_clusters = original['cluster'].unique()
    # dst_external = {}
    arr = []
    w = 0
    for d in external:
        print(w, len(external))
        w += 1
        min_dst = 100000000
        # cl = -1
        for c in list_clusters:
            cluster_data = original.loc[original['cluster'] == c]
            products = cluster_data['values'].values
            dst = [euclidean(d, i) for i in products]
            mean_dst = sum(dst)/len(dst)
            if mean_dst < min_dst:
                min_dst = mean_dst
                # cl = c
        # dst_external[d] = (min_dst, cl)
        arr.append(min_dst)

    return sum(arr)/len(arr)

def preprocess_original():
    init_data = pd.read_csv('../../data/transformations_k2g.csv', delimiter=";")
    init_data["product_name"] = init_data["grocery"]

    transformation = pd.read_csv("..//..//data//groceries_transformation.csv")

    data = pd.merge(transformation, init_data, on=["product_name", "product_name"])
    data = data.drop(columns=["name_1_w", "product_id", "grocery", "Unnamed: 0_x", "Unnamed: 0_y"])
    data = data.sort_values(by="order_id")

    product_names = data["product_name"].unique()
    product_names_t = data["transformed"].unique()
    orders = data['order_id'].unique()

    terms, sentences = get_words(data["transformed"].values)
    data["sentences"] = sentences

    vec_size = 200
    model = Word2Vec.load("../../data/word2vec_original.model")
    voc = list(model.wv.vocab)
    # model = Word2Vec(sentences, min_count=20, compute_loss=True, iter=100, size=vec_size)
    # model.save("../../data/word2vec_original.model")
    print('model is loaded')
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
        print("Pre Step: {}/{}".format(step, len(orders)))
    return matrix_pn

def clustering_word2vec(matrix_pn, k):
    pred = k_means(matrix_pn, k)
    d = pd.DataFrame()
    d['cluster'] = pred
    d['values'] = matrix_pn
    # draw_matrix(matrix_pn, product_names, k, "w2v")
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
    model = Word2Vec.load("../../data/word2vec_external.model")
    # model = Word2Vec(sentences, min_count=20, compute_loss=True, iter=100, size=vec_size)
    # model.save("../../data/word2vec_external.model")
    voc = list(model.wv.vocab)
    orders_id = []

    matrix_pn = []
    step = 0
    for order in orders:
        tmp_data = data.loc[data['order_id']==order]
        tmp_items = tmp_data['transformed'].unique()

        step += 1
        tmp_vec = [0 for _ in range(vec_size)]
        for product in tmp_items:
            for item in product.split(" "):
                if item in voc:
                    tmp_vec = [a + b for a, b in zip(tmp_vec, model[item])]
        matrix_pn.append(tmp_vec)
        orders_id.append(order)

    return matrix_pn, orders_id


def w2v_clustering_distance(matrix, k):

    # original = clustering_word2vec(matrix, k)
    # print('original data are clustered')


    external, orders_external = get_external_w2v()
    print('external model')

    # original = original.sort_values(by='cluster')

    # sil_avg = silhouette_score(original['values'].tolist(), original['cluster'].tolist())
    # sil_vals = silhouette_samples(original['values'].tolist(), original['cluster'].tolist())
    # original['sil_value'] = sil_vals
    # left_border = 0

    # for cluster in list_clusters:
    #     data_to_draw = original.loc[original['cluster'] == cluster]
    #     x_vals = [i for i in range(left_border, left_border+len(data_to_draw))]
    #     left_border = left_border+len(data_to_draw) + 10
    #     plt.bar(x_vals, data_to_draw['sil_value'].tolist())
    #     print('cluster {} - ok'.format(cluster))
    # plt.savefig('C:\\Users\\Tatiana\\Pictures\\clusters\\cluster_{}.png'.format(k))
    # plt.close()
    # print("Mean Silhouette value for {} clusters: {}".format(k, sil_avg))
    # print("Min Silhouette value for {} clusters: {}".format(k, min(original['sil_value'].tolist())))
    # original.to_csv('C:\\Users\\Tatiana\\Pictures\\clusters\\cluster_{}.csv'.format(k))
    original = pd.read_csv('C:\\Users\\Tatiana\\Pictures\\clusters\\cluster_{}.csv'.format(k))
    list_clusters = original['cluster'].unique()
    original_values = [i[1:-1].split(',') for i in original['values'].tolist()]
    original['values'] = original_values
    coeff = []
    vector_order = []
    order_id_integrated = []
    step = 0
    for idx2 in range(len(external)):
        e = external[idx2]
        distances = euclidean_distances(original_values, [e])
        idx = list(distances).index(distances.min())
        closest_cluster = original['cluster'].tolist()[idx]

        original_e = original.loc[original['cluster'] == closest_cluster]
        # original_e_values = [i[1:-1].split(',') for i in original_e['values'].tolist()]

        original_not_e = original.loc[original['cluster'] != closest_cluster]
        # original_not_e_values = [i[1:-1].split(',') for i in original_not_e['values'].tolist()]


        distances_e = euclidean_distances(original_e['values'].tolist(), [e])
        distances_not_e = euclidean_distances(original_not_e['values'].tolist(), [e])

        min_dist_b = distances_not_e.min()
        mean_dist_a = distances_e.mean()
        sil_val_external = (min_dist_b - mean_dist_a)/max(min_dist_b, mean_dist_a)
        print('Step: {}/{}; Cluster: {}; Silhouette: {}'.format(step, len(external),closest_cluster, sil_val_external))
        coeff.append(sil_val_external)
        step += 1
        if sil_val_external > 0:
            order_id_integrated.append(orders_external[idx2])
            vector_order.append(e)
    # plt.bar([i for i in range(len(coeff))], coeff)
    # plt.show()
    # plt.hist(coeff)
    # plt.show()
    coedff_dataframe = pd.DataFrame()
    coedff_dataframe['coeff'] = coeff
    coedff_dataframe['orderr_id'] = orders_external

    data_integrated = pd.DataFrame()
    data_integrated['order_id'] = order_id_integrated
    data_integrated['vals'] = vector_order

    coedff_dataframe.to_csv('coeff_sil.csv')
    data_integrated.to_csv('data_itegrated.csv')




    # dst_external = {}
    # arr = []
    # st = 0
    # for d in external:
    #     st += 1
    #     print(st, "/", len(external))
    #     min_dst = 100000000
    #     # cl = -1
    #     for c in list_clusters:
    #         cluster_data = original.loc[original['cluster'] == c]
    #         products = cluster_data['values'].values
    #         dst = [euclidean(d, i) for i in products]
    #         mean_dst = sum(dst) / len(dst)
    #         if mean_dst < min_dst:
    #             min_dst = mean_dst
    #             # cl = c
    #     # dst_external[d] = (min_dst, cl)
    #     arr.append(min_dst)
    #
    # return sum(arr) / len(arr)

if __name__ == "__main__":
    # tmp_i = []
    # tmp_r = []
    # for i in range(1, 140):
    #     t = cosine_clustering_distance(i)
    #     r = w2v_clustering_distance(i)
    #     print(i, t, r)
    #     tmp_i.append(t)
    #     tmp_r.append(r)
    #
    # plt.plot([i for i in range(1, 140)], tmp_i, label="cos")
    # plt.plot([i for i in range(1, 140)], tmp_r, label='w2v')
    # plt.legend()
    # plt.xlabel("number of clusters")
    # plt.ylabel("Average distance")
    # plt.show()
    # t = cosine_clustering_distance(15)
    # matrix = preprocess_original()
    # for k in range(20, 40):
    matrix = []
    t = w2v_clustering_distance(matrix, 6)
    print(t)
