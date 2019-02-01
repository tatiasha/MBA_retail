import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "..//..//data//dunnhumby"


if __name__ == "__main__":
    transaction_data = pd.read_csv("{}//transaction_data.csv".format(data_path))
    products = pd.read_csv("{}//product.csv".format(data_path))
    data = pd.merge(transaction_data, products, on=["PRODUCT_ID", "PRODUCT_ID"])
    aisles = data["COMMODITY_DESC"].unique()
    data_diff = []
    data_orders = []
    print(data.head())
    weeks = transaction_data["WEEK_NO"].unique()
    for month in range(0, len(weeks), 4):
        print(month, len(weeks))
        transaction_month = pd.DataFrame()
        for u in range(month, month+4):
            if u < len(weeks):
                transaction_month = transaction_month.append(data.loc[data["WEEK_NO"] == weeks[u]])
        tmp_orders_month = []

        for a in aisles:
            aisle_month_data = transaction_month.loc[transaction_month["COMMODITY_DESC"] == a]
            tmp_orders_month.append(len(aisle_month_data))

        if month != 0:
            tmp_diff = [a_i - b_i for a_i, b_i in zip(tmp_orders_month, data_orders[-1])]
            # tmp_div = []
            # for q in range(len(tmp_diff)):
            #     if data_orders[-1][q]:
            #         tmp_div.append(tmp_diff[q]/data_orders[-1][q])
            #     else:
            #         tmp_div.append(tmp_diff[q])
            # data_diff.append([i*100 for i in tmp_div])
            data_diff.append(tmp_diff)
        else:
            data_diff.append([0 for _ in aisles])
        data_orders.append(tmp_orders_month)

    idx = [month/4+1 for month in range(0, len(weeks), 4)]
    d = pd.DataFrame(data_diff, columns=aisles, index=idx)
    r = 0
    result_matrix = [[0 for _ in aisles] for _ in idx]
    result_data = pd.DataFrame(result_matrix, columns=aisles, index=idx)

    for a in aisles:
        result_data[a] = [abs(i)/max(d[a].values) for i in d[a]]
        # plt.plot(idx, d[a])
        # plt.title(a)
        # plt.xlabel("months")
        # plt.ylabel("values")
        # plt.savefig("..//..//data//figs//{}.png".format(r))
        # r += 1
        # plt.close()

    # for i in idx:
    #     val = d.iloc[int(i)].values
    #     print(max(val), val.tolist().index(max(val)), min(val), val.tolist().index(min(val)))


    sns.heatmap(result_data.as_matrix(), cbar=True)
    plt.ylabel("%")
    plt.xlabel("COMMODITY_DESC")
    plt.show()
    # d.to_csv("..//..//data//trends.csv")





