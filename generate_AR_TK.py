import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def concat_clusters(cluster_1, cluster_2):
    return pd.concat([cluster_1, cluster_2])

def get_cluster(file_name):
    cluster = pd.read_csv(file_name)
    cluster = cluster.drop(columns=['Unnamed: 0'])
    return cluster

def get_all_clusters():

    path_train = 'tmp/merge_cluster'
    K1 = get_cluster('{0}_1.csv'.format(path_train))
    K2 = get_cluster('{0}_2.csv'.format(path_train))
    K3 = get_cluster('{0}_3.csv'.format(path_train))
    K4 = get_cluster('{0}_4.csv'.format(path_train))

    path_grocery = 'tmp/groceries_cluster'
    G1 = get_cluster('{0}_1.csv'.format(path_grocery))
    G2 = get_cluster('{0}_2.csv'.format(path_grocery))
    G3 = get_cluster('{0}_3.csv'.format(path_grocery))
    G4 = get_cluster('{0}_4.csv'.format(path_grocery))

    print(len(K1), len(K2), len(K3), len(K4))
    K4 = concat_clusters(K4, G1)
    K3 = concat_clusters(K3, G2)
    K2 = concat_clusters(K2, G3)
    K1 = K1

    K4.to_csv('tmp/K4_table.csv')
    K3.to_csv('tmp/K3_table.csv')
    K2.to_csv('tmp/K2_table.csv')
    K1.to_csv('tmp/K1_table.csv')
    print(len(G1), len(G2), len(G3), len(G4))
    print(len(K1), len(K2), len(K3), len(K4))

    return K1, K2, K3, K4

def generate_rules(cluster, N, support):
    f_i_data = apriori(cluster, min_support=support, use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'], ascending=False)
    f_items = f_i_data['itemsets'].tolist()
    print('fi_ok')
    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'], ascending=False)
    print('rules', len(rules))
    rules.to_csv('tmp/rules_GK_cluster_{0}.csv'.format(N ))


K1, K2, K3, K4 = get_all_clusters()
#print(len(K1), len(K2), len(K3), len(K4))
#generate_rules(K1,1,0.00001)
#generate_rules(K2,2,0.0003)
#generate_rules(K3,3,0.001)
#generate_rules(K4,4,0.0002)
