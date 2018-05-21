import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

flag = 1
if flag == 0:
    conf = pd.read_csv('tmp/confidence_clusters_v1.csv')
    plt.title('confidence_clusters')
else:
    conf = pd.read_csv('tmp/confidence_train_v1.csv')
    plt.title('confidence_train')

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