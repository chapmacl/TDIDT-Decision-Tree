import csv
import pandas as pd
import math
import scipy.stats


def Main():
    
    training_set = pd.read_csv('gene_expression_training.csv')    
    testing_set =  pd.read_csv('gene_expression_test.csv') 
    
    print(training_set)
    set_entropy = Entropy(training_set)
    print(set_entropy)
    
    train_y = training_set['class_label'].copy()
    train_x = training_set.drop(columns = ['class_label'])
    
    entropies = Entropy_columns(training_set)
    print (entropies)
    
def Entropy(df):
    pos = 0
    neg = 0
    
    for row in df['class_label']:
        if (row == 1.0):
            pos = pos + 1
        else:
            neg = neg + 1
    
    entropy = -(pos/(pos+neg))*math.log2(pos/(pos+neg)) - (neg/(pos+neg))*math.log2(neg/(pos+neg))
    return entropy

def Entropy_columns(df):
    entropies = []
    for columns in df:
        p_data = df[columns].value_counts()/len(df)
        entropy = scipy.stats.entropy(p_data)
        entropies.append(entropy)
    return entropies

Main()
