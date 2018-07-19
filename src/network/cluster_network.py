import networkx as nx
import pandas as pd
import community

print("Communities detection(clustering)")
data_path = "E:\Projects\MBA_retail\\tmp"

dataset_path = "E:\Projects\MBA_retail\\tmp\datasets" #for change_train
#dataset_path = "E:\\Data\\bakery\\20000"   #for bakery (20000 orders)

dataset = pd.read_csv('{0}\\change_train.csv'.format(dataset_path))
#dataset = pd.read_csv('{0}\\20000-out2.csv'.format(dataset_path))

lbls = list(dataset)
dataset = dataset.values

#lbls = pd.read_csv('{0}\\EB-build-goods.csv'.format(data_path))  #for bakery only
#lbls = lbls.values  #for bakery only

#name = "bakery_20000"
name = "change_train"
#graph = pd.read_csv('{0}\\train_1000.csv'.format(data_path))
graph = pd.read_csv('{0}\\{1}_graph.csv'.format(data_path, name))

orders = graph.values
network = nx.Graph()
for i in orders:
    #elems = i[0].split(' ') #for bakery
    elems = i[0].split(';') #for change_train

    network.add_edge(str(elems[0]), str(elems[1]), weight=int(elems[2]))

print("Graph is completed with {0} orders".format(nx.number_of_nodes(network)))

# Louvain method
# short description of method https://arxiv.org/pdf/0803.0476.pdf
parts = community.best_partition(network)
values = [parts.get(node) for node in network.nodes()]
com = []

for comm in set(values):
    print("Community {0}/{1} is preparing".format(comm+1, max(set(values))+1))
    for idx, c in parts.items():
        if c == comm:
            com.append(idx)

    result = []
    for i in dataset:
        for j in com:
            if int(i[0]) == int(j):
                result.append(i)

    my_df = pd.DataFrame(result)
    my_df.to_csv('{0}/com{1}_{2}.csv'.format(data_path, comm, name), index=False, header=False)

    orders_com = []
    for i in result:
        tmp = []
        for j in range(1,len(i)-1):
            if i[j] == 1:
                tmp.append(lbls[j])
        if(tmp != []):
            orders_com.append(tmp)

    my_df_lbl = pd.DataFrame(orders_com)
    my_df_lbl.to_csv('{0}/com{1}_{2}_lbl.csv'.format(data_path, comm, name), index=False, header=False)
    print("Community {0}/{1} is completed with {2} orders".format(comm+1, 1+max(set(values)), len(orders_com)))

print("Set of communities", set(values))