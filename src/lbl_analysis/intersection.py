import pandas as pd


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


def parse(lbl):
    return_lbl = []
    for i in lbl:
        t = i[0]
        t = t.replace("[","")
        t = t.replace("]", "")
        t = t.replace(" ", "")
        t = t.replace("'", "")
        t = t.split(',')
        return_lbl.append(t)
    return return_lbl

list_weight = pd.read_csv('tmp/data/list_weight.csv')
data_all_kaggle = pd.read_csv('tmp/data/data_all_kaggle.csv')
data_all_grocery = pd.read_csv('tmp/data/data_all_grocery.csv')
data_words = pd.read_csv('tmp/data/data_words.csv')
data_words = data_words.as_matrix()
lbl_gros = pd.read_csv('tmp/data/lbl_gros.csv')
lbl_gros = lbl_gros.as_matrix()

# print(list_weight.head())
# print("====")
# print(data_all_kaggle.head())
# print("====")
# print(data_all_grocery.head())
# print("====")


#intersection with grocery

print(type(data_words[0][0]))
data_words = parse(data_words)
lbl_gros = parse(lbl_gros)
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
            weight = 0.0
            for l in result:
                t = list_weight.loc[list_weight['item'] == l]
                t = t.as_matrix()
                if len(t) > 0 and lbl1 not in lbl_tmp:
                    weight += float(t[0][2])
                    lbl_weighted = lbl_weighted.append({'array_groc': lbl1,'array_kaggle': lbl2, 'weight': weight, 'grocery_id': lbl_gros.index(lbl1)+2, 'kaggle_id': data_words.index(lbl2)+2}, ignore_index=True)
                    lbl_tmp.append(lbl1)
                #k_id = (data_all_kaggle.loc[data_all_kaggle['array_kaggle'] == lbl2]).as_matrix()
                #gr_id = (data_all_grocery.loc[data_all_grocery['array_groc'] == lbl1]).as_matrix()
                #print(k_id, gr_id)

                lbl_weighted = lbl_weighted.append({'array_groc': lbl1, 'array_kaggle': lbl2, 'weight': weight},
                                                   ignore_index=True)
            lbl_tmp.append(tmp)
            tmp = []

        if result != [] and len(result) > 0 and len(lbl1)==1 and ch:
            weight = 0
            for l in result:
                t = list_weight.loc[list_weight['item'] == l]
                t = t.as_matrix()
                if len(t) > 0 and lbl1 not in lbl_tmp:
                    weight += float(t[0][2])
                    lbl_weighted = lbl_weighted.append({'array_groc': lbl1,'array_kaggle': lbl2, 'weight': weight,'grocery_id': lbl_gros.index(lbl1)+2, 'kaggle_id': data_words.index(lbl2)+2}, ignore_index=True)
                    lbl_tmp.append(lbl1)
            tmp = []

lbl_weighted = lbl_weighted.sort_values(by='weight', ascending=False)

_mt_ = pd.merge(lbl_weighted,data_all_kaggle, on = ['kaggle_id','kaggle_id'])
_mt_ = pd.merge(_mt_,data_all_grocery,on=['grocery_id','grocery_id'])
_mt_ = _mt_.sort_values(by='grocery_id', ascending=False)

print(len(_mt_), len(data_all_kaggle))
_mt_ = _mt_[['product_name', 'product_name_groc']]
data_path = "E:\Data\kaggle"

_mt_.to_csv('{0}/products_v2.csv'.format(data_path))


orders = pd.read_csv('{0}/orders.csv'.format(data_path))
train = pd.read_csv('{0}/order_products__train.csv'.format(data_path))
order_train = pd.merge(train,orders,on=['order_id','order_id'])
products = pd.read_csv('{0}/products.csv'.format(data_path))
products = pd.merge(_mt_,products, on = ['product_name','product_name'])
_mt = pd.merge(train,products, on = ['product_id','product_id'])
data = pd.merge(_mt,orders,on=['order_id','order_id'])

data.to_csv('tmp/data_merge.csv')
data = pd.crosstab(data['order_id'], data['product_name_groc'])
data.to_csv('tmp/change_train.csv')


print(data.head())


