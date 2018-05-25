import pandas as pd
import numpy as np
from copy import copy
import csv

stop_list = ['medium','canned','frozen'," ", ",", "%", "!", ".", "&", "oz", "/", "", "™", "®", "(", ")", ";", "*", "=", ":","@", "+", 'and',]
stop_list+=['instant','baking','other','light','soft','red','spread','sparkling','packaged']
stop_list+=['long', 'life']

exceprion_1 =[['milk', 'berries'], ['yogurt', 'yoghurt']]
exception_2 = [['fat'],['lowfat','low','low-fat']]
exception_3 = [['berries'], ['milk']]
exceptions = []
exceptions.append(exceprion_1)
exceptions.append(exception_2)
exceptions.append(exception_3)

def check_exceptions(lbl1, lbl2):
    first = (set(lbl1))
    second = (set(lbl2))
    #print(first, second)
    ch = True
    for i in exceptions:
        ch_1 = list(first & set(i[0]))
        ch_2 = list(second & set(i[1]))
        if ch_1 != [] and ch_2 != []:
            ch = False
    #ch = ch_1
    return ch


def convert_name(dataset):
    lbl_array = []
    N = len(dataset)
    for i in dataset:
        str_tmp = ''
        tmp_array = []
        for st in i:
            if (st not in stop_list):
                str_tmp += st
            else:
                if (len(str_tmp) > 2 and str_tmp not in stop_list):
                    tmp_array.append(str_tmp.lower())
                str_tmp = ''
        if (str_tmp != '' and str_tmp not in stop_list):
            tmp_array.append(str_tmp.lower())
        lbl_array.append(tmp_array)
        tmp_array = []

    return lbl_array

def readcsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0
    a = []

    for row in reader:
        a.append(row)
        rownum += 1

    ifile.close()
    return a

data_path_groc = 'E:\Data\groceries'
data_gros = readcsv(data_path_groc+"/groceries.csv")

label_gros_tmp = []
for i in data_gros:
    for st in i:
        tmp_str = ''
        for s in st:
            if s != ',':
                tmp_str += s
            else:
                if tmp_str not in label_gros_tmp:
                    label_gros_tmp.append(tmp_str)
                tmp_str = ''
        if tmp_str not in label_gros_tmp:
            label_gros_tmp.append(tmp_str)

lbl_gros = convert_name(label_gros_tmp)

outfile = open('tmp/data/lbl_gros.csv','w')
out = csv.writer(outfile)
out.writerows(map(lambda xi: [xi], lbl_gros))
outfile.close()

cols = ["product_name_groc", "array_groc", "grocery_id"]

data_all_grocery = pd.DataFrame(columns=cols)
for i in range(len(lbl_gros)):
    data_all_grocery = data_all_grocery.append({'product_name_groc': label_gros_tmp[i], 'array_groc': lbl_gros[i], 'grocery_id':i+1}, ignore_index=True)

data_all_grocery.to_csv('tmp/data/data_all_grocery.csv')

data_path = "E:\Data\kaggle"
orders = pd.read_csv('{0}/orders.csv'.format(data_path))
train = pd.read_csv('{0}/order_products__train.csv'.format(data_path))
order_prior = pd.merge(train,orders,on=['order_id','order_id'])
order_prior = order_prior.sort_values(by=['user_id','order_id'])
products = pd.read_csv('{0}/products.csv'.format(data_path))
aisles = pd.read_csv('{0}/aisles.csv'.format(data_path))
_mt = pd.merge(train,products, on = ['product_id','product_id'])
_mt = pd.merge(_mt,orders,on=['order_id','order_id'])
mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])
popular_kaggle = mt['product_name'].value_counts()[0:1000]
popular_kaggle.to_csv('tmp/popular_kaggle.csv')
popular_products = pd.read_csv('tmp/popular_kaggle.csv', names = "nf")
popular_products = popular_products.as_matrix()
popular_products = popular_products[:,0]


col = (list(products))
data_product = pd.DataFrame(columns=col)

for i in popular_products:
    t = products.loc[products['product_name'] == i]
    data_product = data_product.append(t, ignore_index=True)

print(len(data_product))
data_tmp = pd.merge(data_product, aisles, on=['aisle_id', 'aisle_id'])

data = data_tmp[['product_name', 'aisle']]
data = data.as_matrix()

stop_list = [" ", ",", "%", "!", ".", "&", "oz", "/", "", "™", "®", "(", ")", ";", "*", "=", ":","@", "+", 'and']
ends = ['s', 'y', 'd', 'ie', 'ed', 'ly', 'e', 'a', 'ny', 'es', "'n", 'o', 'z', 'os', 'i', 'rs', 'er']
unique_words = []
data_words = []
data_words_tmp = []

#get unique words and data words list
for i in data:
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

outfile = open('tmp/data/data_words.csv','w')
out = csv.writer(outfile)
out.writerows(map(lambda xi: [xi], data_words))
outfile.close()

data_name = data_tmp['product_name'].as_matrix()
cols = ["product_name", "array_kaggle", 'kaggle_id']
data_all_kaggle = pd.DataFrame(columns=cols)
for i in range(len(data_name)):
    data_all_kaggle = data_all_kaggle.append({'product_name': data_name[i], 'array_kaggle': data_words[i], 'kaggle_id': (i+1)}, ignore_index=True)

data_all_kaggle.to_csv('tmp/data/data_all_kaggle.csv')
print("ok - 1")
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

replaces.to_csv('tmp/data/replaces.csv')

print("ok-2")
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

print("ok-3")
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
    #list_weight = list_weight.append({'item': item[i], 'weight': float(weight[i])/sum(weight)}, ignore_index=True)
    if weight[i] != 0:
        list_weight = list_weight.append({'item': item[i], 'weight': 1.0/float(weight[i])}, ignore_index=True)
    else:
        list_weight = list_weight.append({'item': item[i], 'weight': 0}, ignore_index=True)

list_weight = list_weight.sort_values(by='weight', ascending=False)
list_weight.to_csv('tmp/data/list_weight.csv')


print("ok-4")

'''''''''
#intersection with grocery
tmp = []
lbl_tmp = [['array_groc', 'data_words','weight']]
lbl_tmp = []

cols = ["array_groc", "array_kaggle", 'weight', 'kaggle_id', 'grocery_id']
lbl_weighted = pd.DataFrame(columns=cols)
for lbl1 in lbl_gros:
    for lbl2 in data_words:
        result = list(set(lbl1) & set(lbl2))
        ch = check_exceptions(lbl1, lbl2)
        if result!= [] and len(result) > 1 and len(lbl1)>1 and ch:
            weight = 0
            for l in result:
                t = list_weight.loc[list_weight['item'] == l]
                t = t.as_matrix()
                weight += t[0][1]

                k_id = (data_all_kaggle.loc[data_all_kaggle['array_kaggle'] == lbl2]).as_matrix()
                gr_id = (data_all_grocery.loc[data_all_grocery['array_groc'] == lbl1]).as_matrix()
                print(k_id, gr_id)

                lbl_weighted = lbl_weighted.append({'array_groc': lbl1, 'array_kaggle': lbl2, 'weight': weight},
                                                   ignore_index=True)
            lbl_tmp.append(tmp)
            tmp = []
        if result != [] and len(result) > 0 and len(lbl1)==1 and ch:
            weight = 0
            for l in result:
                t = list_weight.loc[list_weight['item'] == l]
                t = t.as_matrix()
                weight += t[0][1]
                lbl_weighted = lbl_weighted.append({'array_groc': lbl1,'array_kaggle': lbl2, 'weight': weight}, ignore_index=True)
            lbl_tmp.append(tmp)
            tmp = []

lbl_weighted = lbl_weighted.sort_values(by='weight', ascending=False)
lbl_weighted.to_csv('tmp/lbl_weighted.csv')


data_all_kaggle.to_csv('tmp/data_all_kaggle.csv')
data_all_grocery.to_csv('tmp/data_all_grocery.csv')

lbl_weighted = pd.read_csv('tmp/lbl_weighted.csv')
lbl_weighted_ = lbl_weighted.as_matrix()
lbls = []
lnl = lbl_weighted[['array_kaggle']].as_matrix()
lnl = np.unique(lnl)
cols = ['array_groc', 'array_kaggle']
list_weight = pd.DataFrame(columns=cols)
for i in lnl:
    c = lbl_weighted.loc[lbl_weighted['array_kaggle'] == i]
    c = c.as_matrix()
    if len(c)>0:
        list_weight = list_weight.append({'array_groc': c[0][1], 'array_kaggle': c[0][2]}, ignore_index=True)

lbl_weighted = list_weight

data_all_kaggle = pd.read_csv('tmp/data_all_kaggle.csv')
data_all_kaggle = data_all_kaggle[['array_kaggle','product_name']]
data_all_grocery = pd.read_csv('tmp/data_all_grocery.csv')



print( lbl_weighted.head())
print("=====")
print(data_all_kaggle.head())

_mt_ = pd.merge(data_all_grocery,data_all_kaggle, on = ['array_groc','array_groc'])
print(len(_mt_), len(lbl_weighted), len(data_all_grocery))


_mt_ = pd.merge(_mt_,data_all_grocery,on=['array_groc','array_groc'])
print(len(_mt_), len(data_all_grocery))

_mt_ = _mt_[['product_name', 'product_name_groc', 'grocery_id']]
print(len(_mt_), len(data_all_kaggle))
print(_mt_.head())
#lbl_weighted.to_csv('tmp/to_change_V3.csv')


# myFile = open('{0}/tmp/intersection.csv'.format(data_path), 'w')
# with myFile:
#     writer = csv.writer(myFile)
#     writer.writerows(lbl_tmp)
print("ok - 5")




#a = (data_tmp['product_name'].value_counts())
#b =  data_tmp[['product_name', 'aisle']]
#b.to_csv('tmp/com.csv')
'''''