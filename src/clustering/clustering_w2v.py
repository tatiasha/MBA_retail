import pandas as pd
from sklearn.cluster import KMeans
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
import sys
import scipy

def get_grocery_orders_matrix(model, orders, all_products):
    matrix_orders = []
    all_products = all_products.drop(columns="Unnamed: 0")
    voc = list(model.wv.vocab)
    orders = orders.values
    print(type(orders))
    size_v = len(model[voc[0]])
    for order in orders:
        order_vector = [0 for _ in range(size_v)]
        products = order[0].split(",")
        elem = []
        for product in products:
            words = all_products.loc[all_products["product_name"] == product]["transformed"].tolist()[0].split(" ")
            for word in words:
                if word in voc and word not in elem:
                    elem.append(word)
                    order_vector += model[word]
        matrix_orders.append(order_vector)
    return matrix_orders


def get_kaggle_orders_matrix(model, orders, grocery_transformation):
    voc = list(model.wv.vocab)
    matrix_orders = []
    orders = orders.drop(columns="Unnamed: 0")
    orders = orders.loc[orders["name_1_w"] > 0.41].sort_values(by="order_id")
    list_orders = list(set(orders["order_id"].tolist()))
    size_v = len(model[voc[0]])
    step = 0
    for n_order in list_orders:
        step += 1
        print("kaggle matrix: {0}/{1}".format(step, len(list_orders)))
        order_vector = [0 for _ in range(size_v)]
        order = orders.loc[orders["order_id"] == n_order]["grocery"].tolist()
        elem = []
        for product in order:
            if grocery_transformation.loc[grocery_transformation["product_name"] == product]["transformed"].tolist():
                words = grocery_transformation.loc[
                    grocery_transformation["product_name"] == product]["transformed"].tolist()[0].split(" ")
                for word in words:
                    if word in voc and word not in elem:
                        elem.append(word)
                        order_vector += model[word]
        matrix_orders.append(order_vector)
    return matrix_orders, list_orders


def filter_orders(grocery, pred, kaggle, orders):
    clusters = list(set(pred))
    data = pd.DataFrame()
    data["order"] = grocery
    data["class"] = pred
    distance_kaggle = []
    cluster_kaggle = []
    idx = []
    step = 0
    for k in kaggle:
        step += 1
        print("filter orders: {0}/{1}".format(step, len(kaggle)))
        mi = sys.maxsize
        c_k = -10
        for c in clusters:
            cl = data.loc[data["class"] == c]["order"].tolist()
            av = (sum(scipy.spatial.distance.cdist(cl, [k]))/len(cl))[0]
            if av < mi:
                mi = av
                c_k = c
        # idx.append(orders)
        distance_kaggle.append(av)
        cluster_kaggle.append(c_k)

    clusterization = pd.DataFrame()
    # clusterization["order"] = orders
    clusterization["distance"] = distance_kaggle
    clusterization["cluster"] = cluster_kaggle
    clusterization.to_csv("../../data/grocery_kaggle_clusterization.csv")

    return 0



if __name__ == "__main__":
    transformations = pd.read_csv("../../data/transformations_k2g.csv", delimiter=";")
    grocery = pd.read_csv("../../data/groceries/groceries.csv", delimiter=";", header=None)
    grocery_products = pd.read_csv("../../data/groceries_transformation.csv")
    model = Word2Vec.load('../../data/model')
    grocery_orders_matrix = get_grocery_orders_matrix(model, grocery, grocery_products)
    print("Grocery matrix is prepared")

    kaggle_orders = pd.read_csv("../../data/transformations_k2g.csv", delimiter=";").sort_values(by="order_id")
    kaggle_orders_matrix, orders = get_kaggle_orders_matrix(model, kaggle_orders, grocery_products)
    print("Kaggle matrix is prepared")


    N_clusters = 4
    clusterer = KMeans(n_clusters=N_clusters).fit(kaggle_orders_matrix)
    centers = clusterer.cluster_centers_
    c_preds = clusterer.predict(kaggle_orders_matrix)
    print(c_preds)
    a = filter_orders(kaggle_orders_matrix, c_preds, grocery_orders_matrix, orders)
