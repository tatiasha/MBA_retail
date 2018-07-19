import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
data_path = "E:\Projects\MBA_retail\\tmp"
orders = pd.read_csv('{0}/datasets\change_train.csv'.format(data_path))
#orders = pd.read_csv('E:\\Data\\bakery\\20000\\20000-out2.csv')
orders = orders.as_matrix()
number_of_orders = 1000#len(orders)
number_of_products = len(orders[0])
network = nx.Graph()
w = []
for i in range(number_of_orders):
    print(i, '/', number_of_orders)
    for j in range(i+1, number_of_orders):
        weight_0 = 0
        weight_1 = 0

        for p in range(1, number_of_products):
            #if orders[i][p] == orders[j][p] == 0:
                #weight_0 += 1
            if orders[i][p] == orders[j][p] == 1:
                weight_1 += 1
        if weight_1 != 0:
            weight = weight_0+weight_1
            network.add_edge(orders[i][0], orders[j][0], weight=weight)
            w.append(weight)

plt.hist(w)
plt.show()

nx.write_weighted_edgelist(network, '{0}//train_1000.edgelist'.format(data_path))