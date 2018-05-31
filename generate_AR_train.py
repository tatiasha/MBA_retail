import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



if __name__ == "__main__":

    # data_path = "D:\Data\\retail\kaggle"
    data_path = "E:\Data\kaggle"
    # data_train = pd.read_csv('{0}/order_products__train.csv'.format(data_path))
    # data_order = pd.read_csv('{0}/orders.csv'.format(data_path))
    # data_product = pd.read_csv('{0}/products.csv'.format(data_path))
    # data_aisle = pd.read_csv('{0}/aisles.csv'.format(data_path))
    #
    # data_tmp = pd.merge(data_train, data_order, on=['order_id', 'order_id'])
    # data_tmp = pd.merge(data_tmp, data_product, on=['product_id', 'product_id'])
    # data_tmp = pd.merge(data_tmp, data_aisle, on=['aisle_id', 'aisle_id'])
    # data_tmp = data_tmp.sort_values(by=['aisle_id'])
    #
    # data_lbl = data_tmp[['aisle_id', 'aisle']]
    #
    # data = pd.crosstab(data_tmp['order_id'], data_tmp['aisle'])
    data = pd.read_csv('tmp/cleaned_groceries.csv')
    print('files ok')
    f_i_data = apriori(data, min_support=0.1, use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    print('fi ok')
    f_items = f_i_data['itemsets'].tolist()
    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    #rules.to_csv('rules_train.csv')
    # rules.to_csv('tmp/rules_train.csv')
    rules.to_csv('tmp/cleaned_groceries_rules.csv')

    print('rules - ', len(rules))