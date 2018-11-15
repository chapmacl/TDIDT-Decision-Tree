import csv
import pandas as pd
import math
import scipy.stats
from _operator import pos, neg


set_entropy = 0

def Main():
    
    training_set = pd.read_csv('gene_expression_training.csv')    
    testing_set =  pd.read_csv('gene_expression_test.csv') 
    
    print(training_set)
    set_entropy = Entropy_set(training_set)
    print(set_entropy)
    
    train_y = training_set['class_label'].copy()
    train_x = training_set.drop(columns = ['class_label'])
    entropies = Entropy_columns(train_x, train_y)
    print (entropies)
        
def Entropy_set(df):
    pos = 0
    neg = 0
    
    for row in df['class_label']:
        if (row == 1.0):
            pos = pos + 1
        else:
            neg = neg + 1
    print(pos)
    print(neg)
    
    entropy = -(pos/(pos+neg))*math.log2(pos/(pos+neg)) - (neg/(pos+neg))*math.log2(neg/(pos+neg))
    return entropy

def Entropy_variables(pos, neg):
    if (pos == 0 or neg == 0):
        entropy = 0
    else:
        entropy = -(pos/(pos+neg))*math.log2(pos/(pos+neg)) - (neg/(pos+neg))*math.log2(neg/(pos+neg))
    return entropy

def Entropy_columns(df, labels):
    entropies = []
    information_gain = []
    #parse columns
    for columns in df:
        sum = 0
        list = pd.concat([df[columns],labels], axis=1)
        list = list.sort_values(by = [columns]).values
        #generate list of unique values in the column
        p_data = df[columns].unique()
        p_data.sort()
        entropy_sum = 0
        for row in p_data:
            #array to hold negative and postive values
            count = [0,0]
            #counts the values and adds how many exist and in what class, since the list is sorted it can break early
            for values in list:
                if (row == values[0] and values[1]==1.0):
                    count[1]+=1
                elif (row == values[0] and values[1]==0.0):
                    count[0]+=1
                elif (values[0]>row):
                    break
            ent = Entropy_variables(count[1], count[0])
            sum = ent*((count[0]+count[1])/len(df[columns]))
            entropy_sum += sum
        entropies.append(entropy_sum)
    return entropies

Main()
