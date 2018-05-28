import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

flag = 2
if flag == 0:
    conf = pd.read_csv('tmp/confidence_clusters_v1.csv')
    plt.title('confidence_clusters')
if flag == 1:
    conf = pd.read_csv('tmp/confidence_train_v1.csv')
    plt.title('confidence_train')
if flag == 2:
    conf = pd.read_csv('tmp/confidence_merge.csv')
    plt.title('Altered train without clusters')
if flag == 3:
    conf = pd.read_csv('tmp/confidence_merge_clusters.csv')
    plt.title('Altered train with clusters')
if flag == 4:
    conf = pd.read_csv('tmp/confidence_GK_clusters.csv')
    plt.title('Altered train and grocery with clusters')

conf = np.array(conf)
# print("RESULT = ", sum / float(C))  # sum -> counter
# print("Zeros", c_zeroz, '/', C)
# counter = Counter(conf)
# X = counter.values()
# Y = counter.keys()
plt.hist(conf)
plt.xlabel('confidence')
plt.ylabel('frequency')
plt.show()