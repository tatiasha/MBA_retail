import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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


def to_set(n, array):
    q = array[:n]
    q = q.tolist()
    q = [i[0] for i in q]
    return set(q)


def frequency(season):
    s_q = season['PRODUCT_ID'].value_counts()
    s_q = s_q.to_frame()
    s_q["FREQ"] = s_q["PRODUCT_ID"]
    s_q["PRODUCT_ID"] = s_q.index
    s_q = pd.merge(s_q, products, on=['PRODUCT_ID', 'PRODUCT_ID'])
    s_q = s_q[["PRODUCT_ID", "FREQ", "COMMODITY_DESC", "SUB_COMMODITY_DESC"]]
    return s_q


if __name__ == "__main__":
    data = get_data()
    vals = data['PRODUCT_ID'].value_counts().to_frame()
    vals["Fre"] = vals.index
    products = pd.DataFrame()
    products['PRODUCT_ID'] = vals["Fre"]
    products['Frequency'] = vals["PRODUCT_ID"]
    products = products.loc[products['Frequency'] > 500]
    data = pd.merge(products, data, on=['PRODUCT_ID', 'PRODUCT_ID'])
    table_data = pd.crosstab(data['BASKET_ID'], data['PRODUCT_ID'])
    table_data['BASKET_ID'] = table_data.index
    data = data[['BASKET_ID', 'DAY']]
    data = pd.merge(table_data, data, on=['BASKET_ID', 'BASKET_ID'])

    print("data - table", len(table_data), len(list(table_data)))
    print(len(data['BASKET_ID'].unique()))
    month_size = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    spring = data.loc[data["DAY"] % 365 > sum(month_size[:2])]
    spring = spring.loc[spring["DAY"] % 365 <= sum(month_size[:5])]

    print("spring - data", len(spring['BASKET_ID'].unique()))

    table_spring = spring.drop_duplicates()#pd.crosstab(spring['BASKET_ID'], data['PRODUCT_ID'])
    print("spring - table", len(table_spring), len(list(table_spring)))
    table_spring.to_csv("E:\Projects\MBA_retail\\tmp\\table_spring_d.csv")
    # frequent_itemsets_spring = apriori(table_spring, min_support=0.0001, use_colnames=True)
    # print("spring - fi")
    # rules_spring = association_rules(frequent_itemsets_spring)
    # rules_spring.to_csv("E:\Projects\MBA_retail\\tmp\\dunnhumby_rules_spring.csv")
    # print("spring - rules", len(rules_spring))

    summer = data.loc[data["DAY"]%365 > sum(month_size[:5])]
    summer = summer.loc[summer["DAY"]%365 <= sum(month_size[:8])]
    print("summer - data", len(summer['BASKET_ID'].unique()))

    table_summer = summer.drop_duplicates()#pd.crosstab(summer['BASKET_ID'], data['PRODUCT_ID'])
    print("summer - table", len(table_summer), len(list(table_summer)))
    table_summer.to_csv("E:\Projects\MBA_retail\\tmp\\table_summer_d.csv")
    # frequent_itemsets_summer = apriori(table_summer, min_support=0.0001, use_colnames=True)
    # print("summer - fi")
    # rules_summer = association_rules(frequent_itemsets_summer)
    # rules_summer.to_csv("E:\Projects\MBA_retail\\tmp\\dunnhumby_rules_summer.csv")
    # print("summer - rules", len(rules_summer))

    autumn = data.loc[data["DAY"] % 365 > sum(month_size[:8])]
    autumn = autumn.loc[autumn["DAY"] % 365 <= sum(month_size[:11])]
    print("autumn - data", len(autumn['BASKET_ID'].unique()))

    table_autumn = autumn.drop_duplicates()#pd.crosstab(autumn['BASKET_ID'], data['PRODUCT_ID'])
    print("autumn - table", len(table_autumn), len(list(table_autumn)))
    table_autumn.to_csv("E:\Projects\MBA_retail\\tmp\\table_autumn_d.csv")
    # frequent_itemsets_autumn = apriori(table_autumn, min_support=0.0001, use_colnames=True)
    # print("autumn - fi")
    # rules_autumn = association_rules(frequent_itemsets_autumn)
    # rules_autumn.to_csv("E:\Projects\MBA_retail\\tmp\\dunnhumby_rules_autumn.csv")
    # print("autumn - rules", len(rules_autumn))

    winter_ = data.loc[data["DAY"] % 365 > sum(month_size[:11])]
    winter = data.loc[data["DAY"] % 365 <= sum((month_size[:2]))]
    winter = winter.append(winter_)
    print("winter - data", len(winter['BASKET_ID'].unique()))

    table_winter = winter.drop_duplicates()#pd.crosstab(winter['BASKET_ID'], data['PRODUCT_ID'])
    print("winter - table ", len(table_winter), len(list(table_winter)))
    table_winter.to_csv("E:\Projects\MBA_retail\\tmp\\table_winter_d.csv")

    # frequent_itemsets_winter = apriori(table_winter, min_support=0.0001, use_colnames=True)
    # print("winter - fi")
    # rules_winter = association_rules(frequent_itemsets_winter)
    # rules_winter.to_csv("E:\Projects\MBA_retail\\tmp\\dunnhumby_rules_winter.csv")
    # print("winter - rules", len(rules_winter))

    # winter = winter["BASKET_ID"].unique()

    # win = to_set(50, w_q)
    # aut = to_set(50, a_q)
    # summ = to_set(50, ss_q)
    # spr = to_set(50, s_q)
    # common = win & aut & summ & spr
    # print(common)
    # print("Winter")
    # print(win - common)
    #
    # print("Spring")
    # print(spr - common)
    #
    # print("Summer")
    # print(summ - common)
    #
    # print("Autumn")
    # print(aut - common)
############################################################################
    # q = set(q)
    # print(len(s_q))

    #print(s_q & ss_q)#&set(a_q)&set(w_q))

    # spring2 = data.loc[data["DAY"] > sum(month_size[:2])]
    # spring2 = spring2.loc[spring2["DAY"] <= sum(month_size[:5])]
    # spring2 = spring2["BASKET_ID"].unique()


    # summer2 = data.loc[data["DAY"] > sum(month_size[:5])]
    # summer2 = summer2.loc[summer2["DAY"] <= sum(month_size[:8])]
    # summer2 = summer2["BASKET_ID"].unique()

    # autumn2 = data.loc[data["DAY"] > sum(month_size[:8])]
    # autumn2 = autumn2.loc[autumn2["DAY"] <= sum(month_size[:11])]
    # autumn2 = autumn2["BASKET_ID"].unique()

    # winter_2= data.loc[data["DAY"] > sum(month_size[:11])]
    # winter2 = data.loc[data["DAY"] <= sum((month_size[:2]))]
    # winter2 = winter2.append(winter_)
    # winter2 = winter2["BASKET_ID"].unique()

    # x = [len(winter), len(spring), len(summer), len(autumn)]
    # x2 = [len(winter2), len(spring2), len(summer2), len(autumn2)]

    # y = ["Winter", "Spring", "Summer", "Autumn"]
    # plt.bar(y,x, label = "2 years")
    # plt.bar(y,x2, label = "1st year")
    # plt.legend()
    # plt.ylabel("Number of orders")
    # plt.show()

    # print(len(winter))
    # print(len(spring))
    # print(len(summer))
    # print(len(autumn))
    # print()
    # print(len(data))
    # print(sum(x))
    # print(len(data) - (len(winter)+len(summer)+len(spring)+len(autumn)))
    # t = [100*x2[i]/x[i] for i in range(len(x))]
    # print(t)
