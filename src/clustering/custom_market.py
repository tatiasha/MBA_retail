import numpy as np
import numpy.random as rnd

rnd.seed(666)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.cluster import SpectralClustering, KMeans, spectral_clustering


def association_rule(rec, lbl, support):
    df = pd.DataFrame.from_records(rec, columns=lbl)
    frequent_itemsets = apriori(df, min_support=support, use_colnames=False)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=support)
    return rules, frequent_itemsets


labels = ['a', 'b', 'c']
P = 3
R = 100
data = rnd.randint(2, size=(R, P))
for i in range(10):
    data[i, 0] = 1
    data[R-1-i, 2] = 1

supp_thr = 0.01

rules, freq_sets = association_rule(data, labels, supp_thr)
print(data[0])
print(data[1])
print("   ")
print(freq_sets)
print(rules)

def dist(a, b):
    return np.linalg.norm(a-b)

class dist_evaluator:
    def __init__(self, fs):
        self.fs = fs

    def dist(self, a, b):
        a_t = np.arange(P)
        a_t = a_t[np.where(a[a_t] > 0)]
        b_t = np.arange(P)
        b_t = b_t[np.where(b[b_t] > 0)]
        distance = 0.0
        counter = 0
        for f in self.fs.values:
            a_check = np.all(np.in1d(f[1], a_t))
            b_check = np.all(np.in1d(f[1], b_t))
            if a_check != b_check:
                distance += f[0]
                counter += 1
        return distance / counter

evaluator = dist_evaluator(freq_sets)

test_distance = evaluator.dist(data[0], data[1])

c_model = SpectralClustering(affinity=evaluator.dist)
clustered = c_model.fit_predict(data)
# clust.fit(data)

pass
