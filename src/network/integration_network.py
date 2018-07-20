import networkx as nx
import pandas as pd
import community

print("Integration of external dataset")
data_path = "E:\Projects\MBA_retail\\tmp"

dataset_path = "E:\Projects\MBA_retail\\tmp\datasets" #for change_train

dataset = pd.read_csv('{0}\\change_train.csv'.format(dataset_path))
number_of_orders = len(dataset)
lbls = list(dataset)
dataset = dataset.values



name = "change_train"
graph = pd.read_csv('{0}\\{1}_graph.csv'.format(data_path, name))

orders = graph.values
network = nx.Graph()
for i in orders:
    elems = i[0].split(';') #for change_train
    network.add_edge(str(elems[0]), str(elems[1]), weight=int(elems[2]))

print("Graph is completed with {0} orders".format(nx.number_of_nodes(network)))
parts = community.best_partition(network)
values = [parts.get(node) for node in network.nodes()]
print('Communities are prepared')

external_dataset = pd.read_csv('{0}\\changed_groc.csv'.format(dataset_path))
external_dataset = external_dataset.values
number_of_ex_orders = len(external_dataset)
number_of_products = len(external_dataset[0])
m = dataset[len(dataset)-1][0]
#average_clust = nx.average_clustering(network)

N_edges_L = nx.number_of_edges(network)
N_nodes_N = nx.number_of_nodes(network)
average_degree_k = float(N_edges_L)/ float(N_nodes_N)
print("Average degree", average_degree_k)
counter_nodes = 0
for j in range(number_of_ex_orders):
    print(j+1, '/', number_of_ex_orders)
    counter_edges = 0
    for i in range(number_of_orders):
        weight_0 = 0
        weight_1 = 0
        for p in range(1, number_of_products):
            #if dataset[i][p] == external_dataset[j][p] == 0:
                #weight_0 += 1
            if dataset[i][p] == external_dataset[j][p] == 1:
                weight_1 += 1
        if weight_1 != 0:
            weight = weight_0+weight_1
            network.add_edge(dataset[i][0], m+external_dataset[j][0], weight=weight)
            counter_edges += 1
            N_edges_L = nx.number_of_edges(network)
            N_nodes_N = nx.number_of_nodes(network)
            average_degree = float(N_edges_L) / float(N_nodes_N)
            if average_degree_k - average_degree < 0.007:
                network.remove_edge(dataset[i][0], m + external_dataset[j][0])
                counter_edges -= 1
    if counter_edges != 0:
        counter_nodes += 1
    print(counter_nodes)

print(counter_nodes, 'nodes are integrated')
integrated_dataset = []
for i in list(network.nodes):
    if i <= m:
        integrated_dataset.append(dataset[i])
    if i > m:
        integrated_dataset.append(external_dataset[i-m])

my_df = pd.DataFrame(integrated_dataset)
my_df.to_csv('{0}/integrated_network.csv'.format(data_path), index=False, header=False)