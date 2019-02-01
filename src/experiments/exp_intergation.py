import pandas as pd
import matplotlib.pyplot as plt


def get_original_data():
    return None


def get_external_data():
    return None


def based_on_product(original_data, external_data):
    return None


def based_on_orders(original_data, external_data):
    w2v = pd.read_csv("../../data/grocery_kaggle_clusterization.csv")
    svd = None
    return w2v, svd

if __name__ == "__main__":
    original_data = get_original_data()
    external_data = get_external_data()

    adapted_data_products_w2v, adapted_data_products_cos = based_on_product(original_data, external_data)

    adapted_data_orders_w2v, adapted_data_orders_svd = based_on_orders(original_data, external_data)
