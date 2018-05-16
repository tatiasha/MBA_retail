import pandas as pd
from mlxtenda.mlxtend.frequent_patterns import apriori
from mlxtenda.mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot as plt
import random
import xlrd
import numpy as np
from sklearn.preprocessing import normalize
import math    
import matplotlib.pyplot as plt
from collections import Counter
def get_products(clients, rules, flag): 
    # flag = 0: return list of recommendation 
    # flag = 1: return confidence
    # flag = 2: return item
    buy = []
    x_data = rules['antecedants'].tolist()
    x_data = [list(_x) for _x in x_data]
    y_data = rules['consequents'].tolist()
    y_data = [list(_y) for _y in y_data]
    N = len(rules)
    recommendation_full = []
    recommendation = []
    prob = []
    for rule in x_data:
        
        if np.array_equal(rule, clients):
            add = y_data[x_data.index(rule)]
            if add not in recommendation_full:
                recommendation_full.append(y_data[x_data.index(rule)])
        
        check = set(rule).issubset(set(clients))
        check_2 = list(set(y_data[x_data.index(rule)]) & set(clients))
        if (check):
            add = list(set(y_data[x_data.index(rule)]) - set(clients))
            if add != []:
                recommendation.append(add)
                prob.append(rules['confidence'][x_data.index(rule)])
    
    col_names =  ['item', 'confidence']
    rec  = pd.DataFrame(columns = col_names)
    for i in range(len(prob)):
        rec = rec.append({'item': recommendation[i],'confidence':prob[i]}, ignore_index=True)
    rec = rec.sort_values(by=['confidence'],ascending = False)
    rec = rec.as_matrix()
    
    selected = []
    selected_2 = 0
    
    if recommendation_full != []:
        selected = recommendation_full
    else:
        if recommendation != []: #!!!!MAX
            #probs = np.array(prob)
            #probs = normalize(probs.reshape(1,-1), 'l1')[0]
            #selected_idx = np.random.choice(len(recommendation), 1, False, probs)[0]
            selected = rec[0][0]#recommendation[selected_idx]
            selected_2 = rec[0][1]#prob[selected_idx]
            
            #if random.random() < probs[selected_idx]:
                #print('!!!!!!!%%%!! ',selected)
                #buy.append(selected)
    recommendation = [item for sublist in recommendation for item in sublist]
    recommendation = list(set(recommendation))
    #recommendation = list((set(recommendation) - set(clients)))
    
    if(flag == 0):
        return recommendation
    if(flag == 1):
        return selected_2
    if(flag == 2):
        return selected    
    
def distCosine (vecA, vecB): 
    def dotProduct (vecA, vecB): 
        d = 0.0
        size_v = len(vecA) 
        for indx in range(size_v): 
            if vecA[indx] in vecB: 
                d += vecA[indx]*vecB[indx] 
        return d
    a =  dotProduct (vecA,vecB)
    b = math.sqrt(dotProduct(vecA,vecA))
    c = math.sqrt(dotProduct(vecB,vecB)) 
    if (b ==0 or c == 0):
        return 0
    else:
        return dotProduct (vecA,vecB) / math.sqrt(dotProduct(vecA,vecA)) / math.sqrt(dotProduct(vecB,vecB)) 

def get_recommendation_cos(Matrix_cos, client, df_lbl):
    recommendation = []
    
    for i in client:
        a = list(Matrix_cos[i])
        m = -1
        for t in a:
            if t > m and t < 1.0:
                m = t
        r = a.index(m)
        #print(i, '->',r)
        if(r != 0):
            y  = df_lbl.set_index(['aisle_id'])
            y = y.loc[r]
            y = y.as_matrix()
            recommendation.append(y[0][0])
            #print(y[0][0])
        #else:
            #recommendation.append('No Recommendation')
            #print('No Recommendation')    
    return recommendation
    
def get_recommendation(client, recommendations, rules):
    
    col_names =  ['antecedants', 'consequents', 'confidence']
    recommendation_rules  = pd.DataFrame(columns = col_names)
    
    x_data = rules['antecedants'].tolist()
    x_data = [list(_x) for _x in x_data]
    y_data = rules['consequents'].tolist()
    y_data = [list(_y) for _y in y_data]
    c_data = rules['confidence'].tolist()
    N = len(x_data)
    for r in range(N):
        ch = list(set(client) & set(x_data[r]))
        ch2 = list(set(recommendations) & set(y_data[r]))
        if(ch != [] and ch2 != []):
            recommendation_rules = recommendation_rules.append({'antecedants': x_data[r],'consequents':y_data[r], 'confidence':c_data[r]}, ignore_index=True)
    #print(recommendation_rules.head())
    result_confidence = get_products(client, recommendation_rules, 1)
    return result_confidence
    
if __name__ == "__main__":
    
    
    data_train = pd.read_csv('../data/dataset3/order_products__train.csv')
    data_order = pd.read_csv('../data/dataset3/orders.csv')
    data_product = pd.read_csv('../data/dataset3/products.csv')
    data_aisle = pd.read_csv('../data/dataset3/aisles.csv')
    
    data_tmp = pd.merge(data_train,data_order,on = ['order_id','order_id'])
    data_tmp = pd.merge(data_tmp,data_product,on = ['product_id','product_id'])
    data_tmp = pd.merge(data_tmp,data_aisle,on = ['aisle_id','aisle_id'])
    data_tmp = data_tmp.sort_values(by=['aisle_id'])
    
    data_lbl = data_tmp[['aisle_id','aisle']]
    
    N_d = 1000
    data = pd.crosstab(data_tmp['order_id'],data_tmp['aisle'])
    #data = data[:N_d]
    data_matrix_ = data.as_matrix()
    data_matrix = data_matrix_[:N_d]
    print("Files are read")
    
    
    df1 = data_tmp[['order_id','aisle','aisle_id']]
    df1 = df1.sort_values(by=['order_id'])
    df = df1.as_matrix()
    
    N = len(data_matrix[0])
    print(len(data_matrix),len(data_matrix[1]))
    
    Matrix_cos = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(i,N):
            i_i = data_matrix[:,i]
            j_j = data_matrix[:,j]
            cosine = distCosine(i_i, j_j)
            Matrix_cos[i][j] = cosine
            Matrix_cos[j][i] = cosine
    print('Matrix_cos - ok')
           
    clients_aisle = []
    clients_aisle_id = []
    tmp_array = [df[0][1]]
    tmp_array_id = [df[0][2]]
    
    N = len(data)
    for i in range(1,N):
        if(df[i][0]==df[i-1][0]):
            tmp_array.append(df[i][1])
            tmp_array_id.append(df[i][2])
            
        else:
            clients_aisle.append(tmp_array)
            clients_aisle_id.append(tmp_array_id)
            
            tmp_array = [df[i][1]]
            tmp_array_id = [df[i][2]]
                
    print('clients - ok')
    
    
    f_i_data = apriori(data, min_support=0.005, use_colnames=True)
    f_i_data = f_i_data.sort_values(by=['support'],ascending=False)    
    
    f_items =  f_i_data['itemsets'].tolist()
    rules = association_rules(f_i_data, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['support'],ascending=False)
    print (rules.head())
    
    #print('len(rules)',len(rules)) 
    print('rules - ok')
    
    counter = 0
    C = 50#len(clients)
    
    conf = []    
    sum = 0
    c_zeroz = 0
    for i in range(C):
        print(i+1,'/', C)
        cos = get_recommendation_cos(Matrix_cos, clients_aisle_id[i],data_lbl)
        cos = list((set(cos) - set(clients_aisle[i])))
        result = get_recommendation(clients_aisle[i],cos,rules)
        
        #compare = clients[i][-1]
        #clients[i] = clients[i][:-1]
        #result = get_products(clients[i], rules)
        if result == 0:
            c_zeroz += 1
        sum += result
        print(result)
        conf.append(result)
        #result = list(set(result[0]))
        #if (compare in result):
            #counter += 1
            #print('!!!')
            #print(compare,';' ,result)
        
        #print(compare,';' ,result)
        
    
    
    #print('client', clients_aisle[0])
    
    #rul = get_products(clients_aisle[0], rules, 0) #list of recommendation
    #print('cos',(cos))
    #print('rules',(rul))
    
    print ("RESULT = ", sum/float(C)) #sum -> counter
    print("Zeros", c_zeroz,'/',C)
    conf = np.sort(conf)
    counter=Counter(conf)
    X = counter.values()
    Y = counter.keys()
    plt.bar(X,Y)
    plt.xlabel('confidence')
    plt.ylabel('frequency')
    plt.show()
    