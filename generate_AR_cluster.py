import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data_path = "E:\Projects\MBA_retail\\tmp\old"
N_clusters = 4

for i in range(N_clusters):
    cluster = pd.read_csv('{0}/train_cluster_{1}.csv'.format(data_path, i+1))
    print(cluster.head())
    f_i_data = apriori(cluster, min_support=0.07, use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)

    f_items = f_i_data['itemsets'].tolist()
    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    print('rules ok', i+1)
    rules.to_csv('tmp/rules_cluster_{0}.csv'.format(i+1))
    print('file ok', i+1)