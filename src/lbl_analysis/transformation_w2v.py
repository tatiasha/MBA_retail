import pandas as pd


if __name__ == "__main__":
    transformations = pd.read_csv("../../data/transformations.csv", sep=";", encoding = "ISO-8859-1")
    orders = pd.read_csv("../../data/kaggle/order_products__train.csv")
    # orders = orders[:5000000]
    products = pd.read_csv("../../data/kaggle/products.csv")
    kaggle_init = pd.merge(products, orders, on=["product_id", "product_id"])
    kaggle_init = kaggle_init.sort_values(by="order_id")
    kaggle_init = kaggle_init[["product_id", "product_name", "order_id"]]
    names = [i.lower() for i in kaggle_init["product_name"].tolist()]
    kaggle_init["product_name"] = names
    print(transformations.head(2))
    print("===============")
    print(orders.head(2))
    print("===============")
    print(products.head(2))
    print("===============")
    print(kaggle_init.head(2))

    res = pd.merge(kaggle_init, transformations, on=["product_name", "product_name"])
    res = res.drop_duplicates()

    print("===============")
    print(res.head(2))
    print("===============")
    print(len(kaggle_init)-len(res))
    a = len(kaggle_init["product_id"].unique())
    b = len(res["product_id"].unique())
    print("PRODUCTS",a, b, a-b)
    print("===============")
    a = len(kaggle_init["order_id"].unique())
    b = len(res["order_id"].unique())
    print("ORDERS",a, b, a - b)
    res.to_csv("../../data/transformations_k2g_train.csv")

