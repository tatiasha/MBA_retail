import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def convert_to_list(a):
    t = [i[1:len(i) - 1] for i in a]
    r = [i.split(', ') for i in t]
    q = [list(map(int, i)) for i in r]
    return q

data_path = "E:\Projects\MBA_retail\\tmp"
global_data_path = "E:\Projects\MBA_retail\\tmp\datasets"
data = pd.read_csv('{0}\\x_target.csv'.format(data_path))
data = data.drop(columns=['Unnamed: 0'])
purchasing = data['Purchasing'].values
target = data['Target'].values

train_dataset = pd.read_csv("{0}\\change_train.csv".format(global_data_path))
train_dataset = train_dataset.drop(columns=['order_id'])
train_dataset = train_dataset.values

X = convert_to_list(purchasing)
y = convert_to_list(target)

p = X[1500:]
r = y[1500:]
X = X[:1500]
y = y[:1500]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 32), random_state=1)
#clf = DecisionTreeClassifier(random_state=2)
clf.fit(X, y)
t = clf.predict(p)
#print(clf.score(X, y))
print("=====")
correct = 0.0

for i in range(len(t)):
    print(np.array(t[i]))
    print(np.array(r[i]))
    if sum(t[i] == r[i]) == 51:
            correct += 1
            print("Ok")
    print("===")

print('Accuracy of the network on the {0} test clients: {1}%'.format(len(t),
                                                                         (100*correct / len(t))))
