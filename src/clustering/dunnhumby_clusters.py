import pandas as pd

def get_clusters(n_clusters):
    data_path = "E:\Data\dunnhumby"
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
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
    df = pd.DataFrame()
    for key in list(clusters_len.keys()):
        df = df.append(clusters[key])
    print("ok - 2")
    clusters_res.update({n_clusters - 1: df})
    return clusters_res
    #for key in


if __name__ == "__main__":
    data_path = "E:\Data\dunnhumby"
    transactions = pd.read_csv('{0}/transaction_data.csv'.format(data_path))
    clients_information = pd.read_csv('{0}/hh_demographic.csv'.format(data_path))
    products_name = pd.read_csv('{0}/product.csv'.format(data_path))
    print(len(products_name))
    total = pd.merge(transactions, clients_information, on=['household_key', 'household_key'])
    total = pd.merge(total, products_name, on=['PRODUCT_ID', 'PRODUCT_ID'])
    ids = list(total['PRODUCT_ID'].unique())
    print(len(ids))

    data_path = 'E:\Projects\MBA_retail\\tmp'
    g = get_clusters(10)
    l = [len(i) for i in g.values()]
    print(l)
    i_c = 0
    for cluster in g.keys():
        i_c += 1
        cluster = g[cluster]
        cluster = cluster[['BASKET_ID', 'PRODUCT_ID']]
        cluster = cluster.sort_values(by=['BASKET_ID'], ascending=False)
        cluster = cluster.values
        orders = [[0 for x in range(len(ids))] for y in range(len(cluster))]
        orders[0][ids.index(cluster[0][1])] = 1
        orders_counter = 0
        print(len(cluster))
        for i in range(1, len(cluster)):
            if cluster[i][0] == cluster[i-1][0]:
                orders[orders_counter][ids.index(cluster[i][1])] = 1
            else:
                orders_counter += 1
                orders[orders_counter][ids.index(cluster[i][1])] = 1

        df = pd.DataFrame(orders, columns=ids)
        df.to_csv("{0}\cluster{1}_dunnhumby.csv".format(data_path, i_c))
    print(len(g), l)
