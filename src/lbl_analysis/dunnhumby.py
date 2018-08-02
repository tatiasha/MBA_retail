import pandas as pd

def get_data():
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


def get_support(total, idx):
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

    segment_statistic = segment_statistic.loc[segment_statistic['Support'] < 1]
    print(segment_statistic.head(20))
    print()
    print()


if __name__ == "__main__":
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

    get_support(total, idx)
