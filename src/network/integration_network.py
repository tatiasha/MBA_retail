import networkx as nx
import pandas as pd
import community
import numpy as np

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
dict_orders_orig = {}
counter_nodes = 0
degrees = []
inf = (nx.info(network))
degrees.append(float(inf[-7:])/2.0)
print(float(inf[-7:]))
for j in range(number_of_ex_orders):
    print(j+1, '/', number_of_ex_orders)
    counter_edges = 0
    for i in range(number_of_orders):
        weight_1 = 0
        weight_1 = sum(dataset[i] & external_dataset[j])

        if weight_1 != 0:
            weight = weight_1
            network.add_edge(dataset[i][0], m+external_dataset[j][0], weight=weight)
        if j == 0:
            dict_orders_orig.update({dataset[i][0]: i})
    inf = (nx.info(network))
    degrees.append(float(inf[83:90]))
    print(degrees[-2], degrees[-1])
    if(abs(degrees[-2] - degrees[-1]) > 2):
        network.remove_node(m + external_dataset[j][0])
        del degrees[-1]
    else:
        counter_nodes += 1

    print(counter_nodes)

print(counter_nodes, 'nodes are integrated')
integrated_dataset = []
cleaned_groc = []
print(dict_orders_orig)
count = 0
ns = list(network.nodes)
N = len(ns)
for i in ns:
    count += 1
    print(count, "/", N)
    i2 = ((dict_orders_orig.get(int(i))))#int(i)
    m = int(m)
    if i2 <= m:
        integrated_dataset.append(dataset[i2])
    if i2 > m:
        integrated_dataset.append(external_dataset[i2-m])
        cleaned_groc.append(external_dataset[i2 - m])

my_df = pd.DataFrame(integrated_dataset)
my_df.to_csv('{0}/integrated_train_network.csv'.format(data_path), index=False, header=False)


my_df2 = pd.DataFrame(cleaned_groc)
my_df2.to_csv('{0}/cleaned_groc_network.csv'.format(data_path), index=False, header=False)