import pandas as pd

def get_statistics():
    data_path = "E:\Data\dunnhumby"
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    for group in list(clients_information)[:-3]:
        names = total[group].unique()
        for segment_name in names:
            segment = total.loc[total[group] == segment_name]
            segment = segment[['PRODUCT_ID', "COMMODITY_DESC", 'SUB_COMMODITY_DESC']]
            print("Segment name:", group + " - " + segment_name)
            print("Size, proportion: {0}%".format(100*len(segment)/len(transactions)))
            statistic = segment['PRODUCT_ID'].value_counts()[10:20]
            statistic = statistic.to_frame()
            statistic['Statistics'] = statistic.index
            statistic = statistic.rename(index=str, columns={"PRODUCT_ID": "Statistics", "Statistics": "PRODUCT_ID"})
            segment_statistic = pd.merge(segment, statistic, on=['PRODUCT_ID', 'PRODUCT_ID'])
            segment_statistic = segment_statistic.drop_duplicates()
            segment_statistic = segment_statistic.sort_values(by=['Statistics'], ascending=False)
            print(segment_statistic.head(10))
            print()
            print()


def get_support():
    data_path = "E:\Data\dunnhumby"
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    idx = []
    for group in list(clients_information)[:-3]:
        print("For group {0} choose subgroup".format(group))
        c = 0
        for sub in total[group].unique():
            print("{0} - {1}".format(c, sub))
            c += 1
        inp = int(input())
        idx.append(inp)
    segment = total
    i = 0
    for group in list(clients_information)[:-3]:
        names = total[group].unique()
        segment_name = names[idx[i]]
        i += 1
        print(group, " - ", segment_name)
        segment = segment.loc[segment[group] == segment_name]
        if(len(segment) == 0):
            print("No clients with that parameters")
            return
        else:
            print("{0} orders".format(len(segment)))

    segment = segment[['PRODUCT_ID', "COMMODITY_DESC", 'SUB_COMMODITY_DESC']]
    statistic = segment['PRODUCT_ID'].value_counts()
    statistic = statistic.to_frame()
    statistic['Statistics'] = statistic.index
    statistic = statistic.rename(index=str, columns={"PRODUCT_ID": "Statistics", "Statistics": "PRODUCT_ID"})

    statistic_total = total['PRODUCT_ID'].value_counts()
    statistic_total = statistic_total.to_frame()
    statistic_total['Statistics'] = statistic_total.index
    statistic_total = statistic_total.rename(index=str, columns={"PRODUCT_ID": "Statistics", "Statistics": "PRODUCT_ID"})

    statistic['Total'] = statistic_total['Statistics']
    statistic['Support'] = statistic['Statistics'] / statistic['Total']

    segment_statistic = pd.merge(segment, statistic, on=['PRODUCT_ID', 'PRODUCT_ID'])
    segment_statistic = segment_statistic.drop_duplicates()
    segment_statistic = segment_statistic.sort_values(by=['Support'], ascending=False)

    #segment_statistic = segment_statistic.loc[segment_statistic['Support'] < 1]
    print(segment_statistic.head(20))
    print()
    print()

def get_dependency():
    data_path = "E:\Data\dunnhumby"
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    group2 = 'HOMEOWNER_DESC'
    group1 = 'INCOME_DESC'
    names1 = total[group1].unique()
    names2 = total[group2].unique()

    for i in names1:
        segment_name1 = i
        segment1 = total.loc[total[group1] == segment_name1]
        for y in names2:
            segment_name2 = y
            print(group1, " - ", segment_name1, ' - ', group2, ' - ', segment_name2)
            segment = segment1.loc[segment1[group2] == segment_name2]
            if len(segment) == 0:
                print("No clients with that parameters")
            else:
                print("{0} orders - {1}".format(len(segment), 100*len(segment) / len(segment1)))
        print()

def get_clusters(n_clusters):
    data_path = "E:\Data\dunnhumby"
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    total = total[:10000]
    groups = ['HOMEOWNER_DESC', 'INCOME_DESC', 'AGE_DESC', 'MARITAL_STATUS_CODE', 'HH_COMP_DESC']
    names = [total[i].unique() for i in groups]
    counter = 0
    clusters = dict()
    clusters_len = dict()

    for i1 in names[0]:
        segment_name1 = i1
        segment1 = total.loc[total[groups[0]] == segment_name1]

        for i2 in names[1]:
            segment_name2 = i2
            segment2 = segment1.loc[segment1[groups[1]] == segment_name2]

            for i3 in names[2]:
                segment_name3 = i3
                segment3 = segment2.loc[segment2[groups[2]] == segment_name3]

                for i4 in names[3]:
                    segment_name4 = i4
                    segment4 = segment3.loc[segment3[groups[3]] == segment_name4]

                    for i5 in names[4]:
                        segment_name5 = i5
                        segment = segment4.loc[segment4[groups[4]] == segment_name5]

                        if len(segment) != 0:
                            print(groups[0], " - ", segment_name1, ' - ', groups[1], ' - ', segment_name2)
                            print(groups[2], " - ", segment_name3, ' - ', groups[3], ' - ', segment_name4)
                            print(groups[4], " - ", segment_name5)
                            print("{0} orders - {1}".format(len(segment), 100*len(segment)/len(total)))
                            print()
                            clusters.update({counter: segment})
                            clusters_len.update({counter: len(segment)})
                            counter += 1

    clusters_res = dict()
    for n_c in range(n_clusters - 1):
        for key in list(clusters_len.keys()):
           if clusters_len[key] == max(clusters_len.values()):
               clusters_res.update({n_c: clusters[key]})
               del clusters_len[key]
    print("ok - 1")
    # df = pd.DataFrame()
    # for key in list(clusters_len.keys()):
    #     df = df.append(clusters[key])
    # print("ok - 2")
    # clusters_res.update({n_clusters - 1: df})
    return clusters_res
    #for key in


if __name__ == "__main__":
    data_path = "E:\Data\dunnhumby"
    # transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    print(clients_information.head())
    print(list(clients_information))
    for i in list(clients_information):
        print(i)
        print(len(clients_information[i].unique()), clients_information[i].unique())
        print('=====')
    # products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    # print(len(products_name))
    # total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    # total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    # ids = list(total['PRODUCT_ID'].unique())
    # print(len(ids))
    #
    # data_path = 'E:\Projects\MBA_retail\\tmp'
    # g = get_clusters(5)
    # l = [len(i) for i in g.values()]
    # print(l)
    # i_c = 0
    # for cluster in g.keys():
    #     i_c += 1
    #     cluster = g[cluster]
    #     cluster = cluster[['BASKET_ID', 'PRODUCT_ID']]
    #     cluster = cluster.sort_values(by=['BASKET_ID'], ascending=False)
    #     cluster = cluster.values
    #     orders = [[0 for x in range(len(ids))] for y in range(len(cluster))]
    #     orders[0][ids.index(cluster[0][1])] = 1
    #     orders_counter = 0
    #     print(len(cluster))
    #     for i in range(1, len(cluster)):
    #         if cluster[i][0] == cluster[i-1][0]:
    #             orders[orders_counter][ids.index(cluster[i][1])] = 1
    #         else:
    #             orders_counter += 1
    #             orders[orders_counter][ids.index(cluster[i][1])] = 1
    #
    #     df = pd.DataFrame(orders, columns=ids)
    #     df.to_csv("{0}\cluster{1}_dunnhumby.csv".format(data_path, i_c))
    # print(len(g), l)
