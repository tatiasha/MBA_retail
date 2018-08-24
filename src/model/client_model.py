import pandas as pd
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data_path = "E:\Data\dunnhumby"


def get_data():
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    return total


def get_history_by_parameters(total, parameters):
    segment = total
    for key in parameters:
        segment = segment.loc[segment[key] == parameters[key]]
    segment = segment.sort_values(by="BASKET_ID")
    segment = segment[["BASKET_ID", "PRODUCT_ID"]].values
    orders = []
    order = [segment[0][1]]
    for i in range(1,len(segment)):
        if segment[i-1][0] == segment[i][0]:
            order.append(segment[i][1])
        else:
            orders.append(order)
            order = [segment[i][0]]
    return segment, orders


def get_parameters(total):
    parameters = dict()
    groups = ['HOMEOWNER_DESC', 'INCOME_DESC', 'AGE_DESC', 'MARITAL_STATUS_CODE', 'HOUSEHOLD_SIZE_DESC']
    names = [total[i].unique() for i in groups]
    for g in range(len(groups)):
        print("From group {0} choose subgroup".format(groups[g]))
        for n in range(len(names[g])):
            print("{0} - {1}".format(n, names[g][n]))
        t = int(input())
        print(t, len(names[g]))
        if (t < -1) or (t >= len(names[g])):
            print("Incorrect value")
            return parameters
        if t != -1:
            parameters.update({groups[g]: names[g][t]})
    print(parameters)
    return parameters


def predict_order(segment, size_of_order):
    #popular = segment["SUB_COMMODITY_DESC"].value_counts()
    products = segment["PRODUCT_ID"].unique()
    products = products[:1000]
    order = []
    for i in range(size_of_order):
        t = random.choice(products)
        while t in order:
            t = random.choice(products)
        order.append(t)
    return order


def convert_to_vector(order, names):
    vec = [0 for i in range(len(names))]
    for o in order:
        vec[names.index(o)] = 1


if __name__ == "__main__":
    data = get_data()
    #parameters = get_parameters(data)
    parameters = dict()
    #parameters.update({'HOMEOWNER_DESC': 'Homeowner', 'INCOME_DESC': '125-149K', 'AGE_DESC':'65+'})
    parameters.update({})
    history, history_orders = get_history_by_parameters(data, parameters)
    print(len(data), len(history))
    table = pd.crosstab(history['BASKET_ID'], history['SUB_COMMODITY_DESC'])
    print(table.head())
    # frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
    # rules = association_rules(frequent_itemsets)
    # print(len(rules))
    #order = predict_order(history, random.randint(2, 7))
    #order_vector = convert_to_vector(order)
