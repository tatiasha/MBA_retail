import pandas as pd
import numpy as np
from copy import copy
data_path = "E:\Data\groceries"

data_path = "D:\D\\retail\kaggle"
data_path = "E:\Data\kaggle"
data_product = pd.read_csv('{0}/products.csv'.format(data_path))
data_aisle = pd.read_csv('{0}/aisles.csv'.format(data_path))
data_tmp = pd.merge(data_product, data_aisle, on=['aisle_id', 'aisle_id'])

data = data_tmp[['product_name', 'aisle']]
data = data.as_matrix()

stop_list = [" ", ",", "%", "!", ".", "&", "oz", "/", "", "™", "®", "(", ")", ";", "*", "=", ":","@", "+", 'and']
ends = ['s', 'y', 'd', 'ie', 'ed', 'ly', 'e', 'a', 'ny', 'es', "'n", 'o', 'z', 'os', 'i', 'rs', 'er']
unique_words = []
data_words = []
data_words_tmp = []

#get unique words and data words list
for i in data[:10]:
    for j in i:
        tmp_str = ""
        for k in j:
            if k not in stop_list and not k.isdigit():
                tmp_str += k
                if (tmp_str == "'"):
                    tmp_str = ""

            else:
                if (tmp_str.lower() not in unique_words) and (tmp_str.lower() not in stop_list)and (len(tmp_str.lower())>2):
                    unique_words.append(tmp_str.lower())
                if (tmp_str.lower() not in stop_list)and (len(tmp_str.lower())>2):
                    data_words_tmp.append(tmp_str.lower())
                tmp_str = ""
        if (tmp_str.lower() not in unique_words) and (tmp_str.lower() not in stop_list) and (len(tmp_str.lower())>2):
            unique_words.append(tmp_str.lower())
        if (tmp_str.lower() not in stop_list) and (len(tmp_str.lower()) > 2):
            data_words_tmp.append(tmp_str.lower())
        tmp_str = ""

    data_words.append(data_words_tmp)
    data_words_tmp = []

#get list of replaces
cols = ["init", "replace"]
replaces = pd.DataFrame(columns=cols)
unique_words_no_repeats = copy(unique_words)
for i in unique_words:
    for j in unique_words:
        if i.find(j)==0 and i!=j and abs(len(i)-len(j))<3:
            l = (len(i)-len(j))
            s = len(i)
            p = i[s-l:s]
            if (p in ends):
                replaces = replaces.append({'init': i, 'replace': j }, ignore_index=True)
                if i in unique_words:
                    loc = unique_words.index(i)
                    unique_words_no_repeats[loc] = j

#replace
data_words_tmp = copy(data_words)
for i in data_words:
    for j in i:
        t = replaces.loc[replaces['init'] == j]
        t = t.as_matrix()
        if len(t) > 0:
            loc_i = data_words.index(i)
            loc = data_words[loc_i].index(j)
            data_words_tmp[loc_i][loc] = t[0][1]

data_tmp = []
for i in data_words_tmp:
    data_tmp.append(list(set(i)))
data_words = copy(data_tmp)


#get weight
item = copy(unique_words_no_repeats)
weight = [0 for i in range(len(item))]
for i in data_words:
    for j in unique_words_no_repeats:
        tmp_count = i.count(j)
        weight[item.index(j)]+=tmp_count
cols = ["item", "weight"]
list_weight = pd.DataFrame(columns=cols)
for i in range(len(item)):
    list_weight = list_weight.append({'item': item[i], 'weight': weight[i]}, ignore_index=True)
list_weight = list_weight.sort_values(by='weight', ascending=False)
print(list_weight.head())


#a = (data_tmp['product_name'].value_counts())
#b =  data_tmp[['product_name', 'aisle']]
#b.to_csv('tmp/com.csv')
