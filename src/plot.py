import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

flag = 10
if flag == 0:
    conf = pd.read_csv('tmp/confidence_clusters_v1.csv')
    plt.title('confidence_clusters')
if flag == 1:
    conf = pd.read_csv('tmp/confidence_train_v1.csv')
    plt.title('confidence_train')
if flag == 2:
    conf = pd.read_csv('tmp/confidence_merge_average.csv')
    plt.title('Altered train without clusters')
if flag == 3:
    conf = pd.read_csv('tmp/confidence_merge_clusters_average.csv')
    plt.title('Altered train with clusters')
if flag == 4:
    conf = pd.read_csv('tmp/confidence_GK_clusters_average.csv')
    plt.title('Altered train and grocery with clusters')
if flag == 5:
    conf = pd.read_csv('tmp/prior_train_statistic.csv')
    plt.title('Train and Prior')
if flag == 6:
    conf = pd.read_csv('tmp/prior_train_statistic_extend.csv')
    plt.title('Extended train and Prior')

#conf = np.array(conf)

# print("RESULT = ", sum / float(C))  # sum -> counter
# print("Zeros", c_zeroz, '/', C)
# counter = Counter(conf)
# X = counter.values()
# Y = counter.keys()
data_path = "E:\Projects\MBA_retail\\tmp"
conf1 = pd.read_csv('{0}/statistics result/train_statistic_5000_v1.csv'.format(data_path)).values
conf2 = pd.read_csv('{0}/train_statistics_network.csv'.format(data_path)).values
plt.hist(conf1, alpha = 0.5, label = 'train', normed=True)
plt.hist(conf2, alpha = 0.5, label = 'extend', normed=True)
plt.legend()
# plt.boxplot([conf1, conf2], labels=['train', 'extend'])
plt.xlabel('')
plt.ylabel('frequency')
plt.show()
c2 = float(sum(conf2))/len(conf2)
c1 = float(sum(conf1))/len(conf1)
print(c2/c1)