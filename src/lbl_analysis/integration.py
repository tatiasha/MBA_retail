import pandas as pd

if __name__ == "__main__":
    external_path = "../../data/kaggle"
    transformation_path = "../../data"
    transformations = pd.read_csv("{0}/transformation_tmp.csv".format(transformation_path))
    transformations = transformations.drop(columns=["Unnamed: 0"])
    transformations = transformations.sort_values(by="close", ascending=False)

    products = pd.read_csv("{0}/products.csv".format(external_path))

    merged_products = transformations.merge(products, left_on='orig_kaggle', right_on='product_name')


    print(merged_products.head())
