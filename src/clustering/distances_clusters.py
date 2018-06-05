import pandas as pd
from scipy.spatial import distance

def get_sum(train_cluster_1):
    train_cluster_1_sum = [0 for i in range(len(train_cluster_1[0]))]
    for i in range(len(train_cluster_1)):
        for j in range(len(train_cluster_1[i])):
            train_cluster_1_sum[j] += train_cluster_1[i][j]
    return train_cluster_1_sum

train_cluster_1 = pd.read_csv('tmp/merge_cluster_1.csv')
train_cluster_1 = train_cluster_1.drop(columns=['Unnamed: 0'])
train_cluster_1 = train_cluster_1.as_matrix()
train_cluster_1_sum = get_sum(train_cluster_1)
print('train_cluster_1')

train_cluster_2 = pd.read_csv('tmp/merge_cluster_2.csv')
train_cluster_2 = train_cluster_2.drop(columns=['Unnamed: 0'])
train_cluster_2 = train_cluster_2.as_matrix()
train_cluster_2_sum = get_sum(train_cluster_2)
print('train_cluster_2')

train_cluster_3 = pd.read_csv('tmp/merge_cluster_3.csv')
train_cluster_3 = train_cluster_3.drop(columns=['Unnamed: 0'])
train_cluster_3 = train_cluster_3.as_matrix()
train_cluster_3_sum = get_sum(train_cluster_3)
print('train_cluster_3')

train_cluster_4 = pd.read_csv('tmp/merge_cluster_4.csv')
train_cluster_4 = train_cluster_4.drop(columns=['Unnamed: 0'])
train_cluster_4 = train_cluster_4.as_matrix()
train_cluster_4_sum = get_sum(train_cluster_4)
print('train_cluster_4')


grocery_cluster_1 = pd.read_csv('tmp/groceries_cluster_1.csv')
grocery_cluster_1 = grocery_cluster_1.drop(columns=['Unnamed: 0'])
grocery_cluster_1 = grocery_cluster_1.as_matrix()
grocery_cluster_1_sum = get_sum(grocery_cluster_1)
print('grocery_cluster_1')

grocery_cluster_2 = pd.read_csv('tmp/groceries_cluster_2.csv')
grocery_cluster_2 = grocery_cluster_2.drop(columns=['Unnamed: 0'])
grocery_cluster_2 = grocery_cluster_2.as_matrix()
grocery_cluster_2_sum = get_sum(grocery_cluster_2)
print('grocery_cluster_2')


grocery_cluster_3 = pd.read_csv('tmp/groceries_cluster_3.csv')
grocery_cluster_3 = grocery_cluster_3.drop(columns=['Unnamed: 0'])
grocery_cluster_3 = grocery_cluster_3.as_matrix()
grocery_cluster_3_sum = get_sum(grocery_cluster_3)
print('grocery_cluster_3')


grocery_cluster_4 = pd.read_csv('tmp/groceries_cluster_4.csv')
grocery_cluster_4 = grocery_cluster_4.drop(columns=['Unnamed: 0'])
grocery_cluster_4 = grocery_cluster_4.as_matrix()
grocery_cluster_4_sum = get_sum(grocery_cluster_4)
print('grocery_cluster_4')


dst_1_1 = distance.euclidean(grocery_cluster_1_sum,train_cluster_1_sum)
dst_1_2 = distance.euclidean(grocery_cluster_1_sum,train_cluster_2_sum)
dst_1_3 = distance.euclidean(grocery_cluster_1_sum,train_cluster_3_sum)
dst_1_4 = distance.euclidean(grocery_cluster_1_sum,train_cluster_4_sum)
print('dst_1_*')

dst_2_1 = distance.euclidean(grocery_cluster_2_sum,train_cluster_1_sum)
dst_2_2 = distance.euclidean(grocery_cluster_2_sum,train_cluster_2_sum)
dst_2_3 = distance.euclidean(grocery_cluster_2_sum,train_cluster_3_sum)
dst_2_4 = distance.euclidean(grocery_cluster_2_sum,train_cluster_4_sum)
print('dst_2_*')

dst_3_1 = distance.euclidean(grocery_cluster_3_sum,train_cluster_1_sum)
dst_3_2 = distance.euclidean(grocery_cluster_3_sum,train_cluster_2_sum)
dst_3_3 = distance.euclidean(grocery_cluster_3_sum,train_cluster_3_sum)
dst_3_4 = distance.euclidean(grocery_cluster_3_sum,train_cluster_4_sum)
print('dst_3_*')

dst_4_1 = distance.euclidean(grocery_cluster_4_sum,train_cluster_1_sum)
dst_4_2 = distance.euclidean(grocery_cluster_4_sum,train_cluster_2_sum)
dst_4_3 = distance.euclidean(grocery_cluster_4_sum,train_cluster_3_sum)
dst_4_4 = distance.euclidean(grocery_cluster_4_sum,train_cluster_4_sum)
print('dst_4_*')

cols = ["name", "G1", 'G2', 'G3', 'G4']
distances = pd.DataFrame(columns=cols)
distances = distances.append({"name":"K1", "G1":dst_1_1, 'G2':dst_2_1, 'G3':dst_3_1, 'G4':dst_4_1}, ignore_index=True)
distances = distances.append({"name":"K2", "G1":dst_1_2, 'G2':dst_2_2, 'G3':dst_3_2, 'G4':dst_4_2}, ignore_index=True)
distances = distances.append({"name":"K3", "G1":dst_1_3, 'G2':dst_2_3, 'G3':dst_3_3, 'G4':dst_4_3}, ignore_index=True)
distances = distances.append({"name":"K4", "G1":dst_1_4, 'G2':dst_2_4, 'G3':dst_3_4, 'G4':dst_4_4}, ignore_index=True)

distances.to_csv('tmp/distances.csv')
print(round(dst_1_1,3), round(dst_1_2,3),round(dst_1_3,3),round(dst_1_4,3))