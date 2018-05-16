import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
conf = pd.read_csv('tmp/confidence.csv')
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