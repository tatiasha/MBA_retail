import pandas as pd
import random
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data_path = "E:\Data\dunnhumby"


def get_data():
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    # transactions = pd.read_csv('E:\Projects\MBA_retail\\tmp\\transactions_dunnhumby.csv'.format(data_path))

    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    total = total.sort_values(by="BASKET_ID")
    # total = total.drop(columns=['Unnamed: 0'])
    return total

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



def param_to_vector(param, total):
    groups = ["DAY", 'HOMEOWNER_DESC', 'INCOME_DESC', 'AGE_DESC', 'MARITAL_STATUS_CODE', 'HOUSEHOLD_SIZE_DESC']
    days = total["DAY"].tolist()
    day = [i % 7 for i in days]
    total["DAY"] = day
    names = [total[i].unique().tolist() for i in groups]
    p = param.drop(index=['household_key', 'PRODUCT_ID', 'BASKET_ID'])
    vec = []
    for g in range(len(groups)):
        vec.append(names[g].index(p[g]))
    return vec


if __name__ == "__main__":
    data = get_data()

    vals = data['PRODUCT_ID'].value_counts().to_frame()
    vals["Fre"] = vals.index
    products = pd.DataFrame()
    products['PRODUCT_ID'] = vals["Fre"]
    products['Frequency'] = vals["PRODUCT_ID"]
    products = products.loc[products['Frequency'] > 500]

    products_transaction = pd.merge(products, data, on=['PRODUCT_ID', 'PRODUCT_ID'])
    print(len(data))
    print(len(products_transaction))
    print(list(products_transaction))
    groups = ["household_key", "PRODUCT_ID", "BASKET_ID", "DAY", 'HOMEOWNER_DESC', 'INCOME_DESC', 'AGE_DESC', 'MARITAL_STATUS_CODE', 'HOUSEHOLD_SIZE_DESC']
    products_transaction = products_transaction[groups]
    products_transaction = products_transaction.sort_values(by="BASKET_ID")
    table = pd.crosstab(products_transaction['BASKET_ID'], products_transaction['PRODUCT_ID'])

    l = len(products_transaction["BASKET_ID"].unique())
    c = 0
    tyu = products_transaction["BASKET_ID"].unique()
    idx = [random.choice(range(len(tyu))) for i in range(int(len(tyu)*0.25))]
    idx = list(set(idx))
    print(len(idx), int(len(tyu)*0.2))

    input_train = []
    input_test = []
    output_train = []
    output_test = []
    basket_train = []
    basket_test = []
    for t in range(len(tyu)):
        c += 1
        print("{0}/{1}".format(c, l))
        tr = products_transaction.loc[products_transaction["BASKET_ID"] == tyu[t]]
        if t in idx:
            input_test.append(param_to_vector(tr.iloc[0], products_transaction))
            basket_test.append(tyu[t])
            tr = table.loc[table.index == tyu[t]]
            output_test.append(tr.values)
        else:
            input_train.append(param_to_vector(tr.iloc[0], products_transaction))
            basket_train.append(tyu[t])
            tr = table.loc[table.index == tyu[t]]
            output_train.append(tr.values)

    train = pd.DataFrame()
    test = pd.DataFrame()
    print(len(idx))
    print(len(idx))

    train["input"] = input_train
    train["target"] = output_train
    train["BASKET_ID"] = basket_train
    train.to_csv("E:\Projects\MBA_retail\\tmp\\train_input_target_dunnhumby.csv")
    print(output_train[:10])


    test["input"] = input_test
    test["target"] = output_test
    test["BASKET_ID"] = basket_test
    test.to_csv("E:\Projects\MBA_retail\\tmp\\test_input_target_dunnhumby.csv")
