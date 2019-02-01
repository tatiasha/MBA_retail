import pandas as pd
import matplotlib.pyplot as plt


def get_reference_data():
    return None


def get_weight_data():
    data = pd.read_csv("../../data/transformations_k2g_train.csv")
    return data


def get_lsi_data():
    data = pd.read_csv("../../data/transformation_lsi.csv")
    return data


def get_w2v_data():
    data = pd.read_csv("..//..//data//weight_unification.csv")
    return data


def comparing(reference_data, compared_data):
    return None


if __name__ == "__main__":
    ex_w = get_weight_data()
    ex_lsi = get_lsi_data()
    ex_w2v = get_w2v_data()

    reference_data = get_reference_data()

    plt.bar(["w2v"], comparing(reference_data, ex_w2v), label="w2v")
    plt.bar(["lsi"], comparing(reference_data, ex_lsi), label="lsi")
    plt.bar(["weight"], comparing(reference_data, ex_w), label="weight")
    plt.legend()
    plt.show()
