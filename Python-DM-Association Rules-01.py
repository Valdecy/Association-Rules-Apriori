############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Association Rules Apriori

# Citation: 
# PEREIRA, V. (2018). Project: Association Rules Apriori, File: Python-DM-Association Rules-01.py, GitHub repository: <https://github.com/Valdecy/Association_Rules_Apriori>

############################################################################

# Required Libraries
import pandas as pd
import numpy as np
from itertools import permutations

def ant_conseq_list(list_names = ["1","2","3"], size = 1): 
    antecedent = ""
    consequent = ""  
    for i in range(0, size):
        antecedent = antecedent + ", " + list_names[i]       
    for i in range(size, len(list_names)):
        consequent = consequent + ", " + list_names[i]    
    return antecedent, consequent

def ant_conseq_freq(Xdata, list_names = ["1","2","3"], min_frk = 1):
    for i in range(len(list_names) - 1, -1, -1):  
        frk = Xdata[list_names[i][0]]   
        for j in range(0, len(list_names[i])): 
            frk = frk*Xdata[list_names[i][j]]     
        if (sum(frk) < min_frk):
            del list_names[i]    
    return list_names

def ant_conseq_support(Xdata, antecedent, consequent):  
    antecedent = antecedent.split()
    consequent = consequent.split()   
    support_A  = Xdata[antecedent[0].replace(",", "")]
    support_C  = Xdata[consequent[0].replace(",", "")]   
    for i in range(0, len(antecedent)):
        support_A = support_A* Xdata[antecedent[i].replace(",", "")]       
    for i in range(0, len(consequent)):
        support_C = support_C* Xdata[consequent[i].replace(",", "")]   
    support_A = sum(support_A)/Xdata.shape[0]
    support_C = sum(support_C)/Xdata.shape[0]   
    return support_A, support_C

def apriori(Xdata, min_freq = 1):    
    min_frk_value = min_freq
    market_basket = list(Xdata.columns.values)
    result = pd.DataFrame(columns = ["Antecedent","Consequent", "Frequence","Suport(Antecedent)","Suport(Consequence)","Suport","Confidence","All Confidence","Cosine", "Lift", "Conviction", "Interestingness"])
    full_list = []
    rpt       = True
    print("********* Starting **********") 
    print("Valid Permutation (n =  2 )")
    list_name = list(permutations(market_basket, 2))
    list_name = ant_conseq_freq(Xdata, list_name, min_frk = min_frk_value)
    full_list.append(list_name)
    count = 3
    
    while rpt:
        temp_list = []
        perm_list = []
        temp_list = list(set([item for sublist in list_name for item in sublist]))       
        if (len(temp_list) >= count):
            for i in range(0, len(list_name)):
                for element in temp_list:
                    if element not in list_name[i]:
                        X_sub = Xdata[list(list_name[i]) + [element]]
                        check = Xdata[list(list_name[i]) + [element]].sum(axis=1)      
                        X_sub = X_sub[check >= X_sub.shape[1]]      
                        if (X_sub.shape[0] >= min_freq):
                            perm_list.append(list(list_name[i]) + [element])          
            #list_name = ant_conseq_freq(Xdata, perm_list, min_frk = min_frk_value)
            print("Valid Permutation (n = ", count,")")
            list_name = perm_list.copy()
            if (len(list_name) > 0):
                full_list.append(list_name)
        else:
            rpt =  False
            print("*********** Done! ***********")            
        count = count + 1
        
    count = 0
    ant = ""
    cot = ""
    
    for k in range(0, len(full_list)):
        for i in range(0, len(full_list[k])):
            columns = len(full_list[k][0])
            stop    = len(full_list[k][0]) - 1
            frq     = Xdata[full_list[k][i][0]]
            rpt     = True
            itens   = 1           
            while rpt:               
                if (columns == 2):
                    ant = full_list[k][i][0]
                    cot = full_list[k][i][1]
                    san = sum(Xdata[full_list[k][i][0]])/Xdata.shape[0]
                    scq = sum(Xdata[full_list[k][i][1]])/Xdata.shape[0]                   
                if (columns > 2):
                    names = ant_conseq_list(list_names = full_list[k][i], size = itens)
                    ant   = names[0] 
                    cot   = names[1] 
                    ant   = ant[2:]
                    cot   = cot[2:]
                    sup   = ant_conseq_support(Xdata, ant, cot)
                    san   = sup[0]
                    scq   = sup[1]                    
                for j in range(0, columns):
                    frq = Xdata[full_list[k][i][j]]*frq                    
                if (sum(frq) > 0 and san > 0 and scq > 0 ):
                    saq = sum(frq)/Xdata.shape[0]
                    con = saq/san
                    acf = saq/max(san, scq)
                    cos = saq/(san*scq)**(1/2)
                    lif = con/scq
                    if (con == 1):
                        cov = float("inf")
                    else:
                        cov = (1-scq)/(1-con)
                    inr = (saq/san)*(saq/scq)*(1 - (saq/Xdata.shape[0]))                   
                stop = stop - 1               
                if (stop == 0):
                    rpt =  False
                result.loc[count] = [ant, cot, sum(frq), san, scq, saq, con, acf, cos, lif, cov, inr]     
                count = count + 1
                ant = ""
                cot = ""
                itens = itens + 1
                print("Rules = ", result.shape[0]) 

    for i in range(0, result.shape[0]):
        for j in range(result.shape[0]-1, 0, -1):
            if (len(result.iloc[i,0].split()) > 1 and len(result.iloc[j,0].split()) > 1 and set(result.iloc[i,0].replace(",", "").split()) == set(result.iloc[j,0].replace(",", "").split()) and i != j):               
                result.iloc[i,0] = "delete"
                result.iloc[i,1] = "delete"                
            if (len(result.iloc[i,1].split()) > 1 and len(result.iloc[j,1].split()) > 1 and set(result.iloc[i,1].replace(",", "").split()) == set(result.iloc[j,1].replace(",", "").split()) and i != j):
                result.iloc[i,0] = "delete"
                result.iloc[i,1] = "delete"           
    result = result[result['Antecedent'] !=  "delete"]
    result = result.reset_index(drop=True)
    print("Unique Rules = ", result.shape[0])
    rule = []    
    for i in range(0, result.shape[0]):
        rule.append("IF " + result.iloc[i,0] + " THEN " + result.iloc[i,1] + " .(support = " + str(round(result.iloc[i,5], 2)) +")")
        rule[i] = rule[i].replace(", ", " AND ")    
    return rule, result

# Function: Get 0-1 Transaction Matrix
def transform_to_0_1_transaction_matrix(Xdata):
    number_of_transactions = Xdata.shape[0]   
    items = []
    for i in range(0, Xdata.shape[0]): 
        for j in range(0, Xdata.shape[1]):
            Xdata.iloc[:,j] = Xdata.iloc[:,j].str.replace(' ', '_')
            if not Xdata.iloc[i,j] in items and pd.isnull(Xdata.iloc[i,j]) == False:
                items.append(Xdata.iloc[i,j])
    matrix_0_1 = pd.DataFrame(np.zeros((number_of_transactions, len(items))))
    matrix_0_1.columns = items
    for i in range(0, matrix_0_1.shape[0]): 
        for j in range(0, matrix_0_1.shape[1]):    
            for k in range(0, Xdata.shape[1]):
                if (Xdata.iloc[i,k]==items[j]):
                    matrix_0_1.iloc[i,j] = 1
    return matrix_0_1

######################## Part 1 - Usage ####################################

# Example 1) With 0-1 Transaction Matrix
df = pd.read_csv('Python-DM-Association Rules-01a.csv', sep = ';')
X = df

a_rules_apriori_1 = apriori(X, min_freq = 3)
a_rules_apriori_1[1].to_csv("apriori.csv", sep = ' ', index = True, header = True)

# Example 2) Without 0-1 Transaction Matrix
df = pd.read_csv('Python-DM-Association Rules-01b.csv', sep = ',', header = None)
Y = transform_to_0_1_transaction_matrix(df)

a_rules_apriori_2 = apriori(Y, min_freq = 4)
a_rules_apriori_2[1].to_csv("apriori.csv", sep = ' ', index = True, header = True)

df = pd.read_csv('Python-DM-Association Rules-01g.txt', sep = '\t')
X = df.iloc[:,1:]

a_rules_apriori = apriori(X, min_freq = 1)
a_rules_apriori[1].to_csv("apriori.csv", sep = ' ', index = True, header = True)
