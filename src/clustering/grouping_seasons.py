import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_path = "/home/niggaplease/Documents/Tatiana/dunnhumby"
data = pd.read_csv("{0}/transaction_data.csv".format(data_path))
products = pd.read_csv("{0}/product.csv".format(data_path))

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

    month_size = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    spring = data.loc[data["DAY"]%365 > sum(month_size[:2])]
    spring = spring.loc[spring["DAY"]%365 <= sum(month_size[:5])]
    s_q = frequency(spring)
    s_q = s_q[["COMMODITY_DESC"]].values
    # spring = spring["BASKET_ID"].unique()


    summer = data.loc[data["DAY"]%365 > sum(month_size[:5])]
    summer = summer.loc[summer["DAY"]%365 <= sum(month_size[:8])]
    ss_q = frequency(summer)
    ss_q = ss_q[["COMMODITY_DESC"]].values
    # summer = summer["BASKET_ID"].unique()


    autumn = data.loc[data["DAY"]%365 > sum(month_size[:8])]
    autumn = autumn.loc[autumn["DAY"]%365 <= sum(month_size[:11])]
    a_q = frequency(autumn)
    a_q = a_q[["COMMODITY_DESC"]].values
    # autumn = autumn["BASKET_ID"].unique()


    winter_ = data.loc[data["DAY"]%365 > sum(month_size[:11])]
    winter = data.loc[data["DAY"]%365 <= sum((month_size[:2]))]
    winter = winter.append(winter_)
    w_q = frequency(winter)
    w_q = w_q[["COMMODITY_DESC"]].values
    # winter = winter["BASKET_ID"].unique()

    win = to_set(50, w_q)
    aut = to_set(50, a_q)
    summ = to_set(50, ss_q)
    spr = to_set(50, s_q) 
    common = win & aut & summ & spr
    print(common)
    print("Winter")
    print(win - common)
   
    print("Spring")
    print(spr - common)

    print("Summer")
    print(summ - common)

    print("Autumn")
    print(aut - common)

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
