import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def grocery_rules():
    grocery = pd.read_csv("../../data/groceries/groceries.csv", delimiter=";", header=None).values.tolist()
    print(grocery)
    n_orders = len(grocery)
    print(n_orders)
    orders_list = []
    product_n = []
    for order in range(n_orders):
        products = grocery[order][0].split(",")
        for product in products:
            orders_list.append(order)
            product_n.append(product)

    data = pd.DataFrame()
    data["order_id"] = orders_list
    data["product_name"] = product_n
    print(data.head())
    data.to_csv('../../data/orders_grocery.csv')
    data = pd.crosstab(data['order_id'], data['product_name'])
    data.to_csv('../../data/table_grocery.csv')

    f_i_data = apriori(data, min_support=0.0005
                                         , use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    f_items = f_i_data['itemsets'].tolist()
    print('fi_ok')

    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    print('rules', len(rules))
    rules.to_csv('../../data/rules_grocery.csv')

def prior_rules():
    prior = pd.read_csv("../../data/transformations_k2g_prior_5m.csv")
    print(prior.head())
    data = pd.crosstab(prior['order_id'], prior['grocery'])
    data.to_csv('../../data/table_prior_5m.csv')

    f_i_data = apriori(data, min_support=0.002
                       , use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    print('fi_ok')

    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    print('rules', len(rules))
    rules.to_csv('../../data/rules_prior_5m.csv')

def adapted_rules():
    # data = pd.read_csv("../../data/kaggle_adapted.csv")
    # print(data.head())
    # print(data["order_id"][0], data["grocery"][0])
    # data = pd.crosstab(data['order_id'], data['grocery'])
    # data.to_csv('../../data/table_kaggle_adapted.csv')
    # print("table ok")
    data = pd.read_csv("../../data/table_train.csv")
    f_i_data = apriori(data, min_support=0.002
                       , use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    print('fi_ok')

    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    print('rules', len(rules))
    rules.to_csv('../../data/rules_train.csv')

    # rules.to_csv('../../data/rules_kaggle_adapted.csv')

if __name__ == "__main__":
    adapted_rules()
    # prior = pd.read_csv("../../data/transformations_k2g_train.csv")
    # print(prior.head())
    # data = pd.crosstab(prior['order_id'], prior['grocery'])
    # data.to_csv('../../data/table_train.csv')
    #
    # f_i_data = apriori(data, min_support=0.004
    #                    , use_colnames=True)
    # f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    # f_items = f_i_data['itemsets'].tolist()
    # print('fi_ok')
    #
    # rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    # rules = rules.sort_values(by=['support'], ascending=False)
    # print('rules', len(rules))
    # rules.to_csv('../../data/rules_train.csv')




