import numpy as np
import copy
import random
import pickle
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import string

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


#reads data from input file and return the list of genes, the values for each and the output for each
def readData(filename):
    attr_value = []
    class_label = []
    fp = open(filename,'r')
    lines = fp.read().split("\n")

    genes = lines[0].split(',')

    for rows in lines[1:]:
        data = rows.split(',')
        attr_value.append(data[:-1])
        class_label.append(data[-1])

    attr_value = attr_value[:-1]
    rez = np.array([[float(attr_value[j][i]) for j in range(len(attr_value))] for i in range(len(attr_value[0]))])

    return genes[:-1], rez, class_label[:-1]


def calcEntropy(matrix):
    entropy = 0

    for i in list(matrix):
        if i!=0:
            entropy += i/float(sum(matrix)) * np.log2(1/(i/float(sum(matrix))))
            #print(i,float(sum(matrix)))

    return entropy

def information_gain(attr_values,class_label):
    values = list(set(attr_values))
    outputs = list(set(class_label))

    matrix = np.zeros((len(values), len(outputs)))

    for i in range(len(attr_values)):
        matrix[values.index(attr_values[i]), outputs.index(class_label[i])] += 1

    h_s = calcEntropy(matrix.sum(axis=0))
    h_s_a = 0

    for i in range(matrix.shape[0]):
        h_s_a += sum(matrix[i,:])/float(len(attr_values))*calcEntropy(matrix[i,:])
        #print(sum(matrix[i,:]),float(len(attrib)))

    return h_s-h_s_a


def bestSplit(attr,label_value):
    unsorted = []
    for i in range(len(attr)):
        unsorted.append((attr[i],label_value[i]))

    sortedData = sorted(unsorted, key=lambda x:x[0])

    sorted_val = [val[0] for val in sortedData]
    sorted_label = [val[1] for val in sortedData]

    split_options = []

    previous_label = sorted_label[0]
    for i in range(1,len(sorted_val)):
        if previous_label != sorted_label[i]:
            discr_att = [0]*i+[1]*(len(sorted_val)-i)
            split_options.append((i, information_gain(discr_att,sorted_label)))
        previous_label = sorted_label[i]

    best_split = max(split_options, key=lambda x:x[1])

    if best_split[0]<=0:
        avg = sorted_val[best_split[0]]
    else:
        avg = (float(sorted_val[best_split[0]-1]) + float(sorted_val[best_split[0]]))/2

    return avg, best_split[1]


def bestAttribute(rez,class_label):
    """
    Calculation of attribute with max Inf_Gain
    :param rez: Data (all columns)
    :param class_label: all values
    :return:
    """
    attrib_info_gain = []
    for col_index in range(len(rez)):
        avg, gain = bestSplit(rez[col_index],class_label)
        attrib_info_gain.append([avg,gain,col_index]) #col_index is the number of column
    best_attribute = max(attrib_info_gain, key=lambda x:x[1])

    return best_attribute


def splitData(best_attribute,data,class_label):
    right_data = []
    left_data = []
    right_label = []
    left_label = []

    for i in range(0, len(data[best_attribute[2]])-1):
        if best_attribute[0] < data[best_attribute[2]][i]:
            right_data.append(data[:, i])
            right_label.append(class_label[i])
        else:
            left_data.append(data[:, i])
            left_label.append(class_label[i])

    return np.array(right_data).transpose(), right_label, np.array(left_data).transpose() , left_label

  
def tdidt(genes,data,class_label,depth,tree):
    if depth >= 3 or len(data) == 0 or class_label.count(0) == len(data) or class_label.count(1) == len(data):
        tree['type'] = 'leaf'
        tree['children'] = []
        tree['value'] = [class_label.count(0), class_label.count(1)]
        tree['label'] = max(set(class_label), key = class_label.count)
        return tree
    else:
        best_attribute = bestAttribute(data,class_label)
        tree['children'] = []
        tree['decision'] = best_attribute[0]
        tree['gain'] = best_attribute[1]
        tree['id'] = best_attribute[2]
        tree['gene'] = genes[best_attribute[2]]
        tree['value'] = [class_label.count(0), class_label.count(1)]

        right_data,right_label,left_data,left_label = splitData(best_attribute,data,class_label)
        #returns one column!!
        tree['children'].append({})
        return tdidt(genes,right_data,right_label,depth+1,tree['children'][-1])
        tree['children'].append({})
        return tdidt(genes,left_data,left_label,depth+1,tree['children'][-1])


genes,data,class_label = readData("gene_expression_training.csv")
#print(label)

print(tdidt(genes,data,class_label,0,{}))

#print(len(genes))
#avg,best = best_split(attr, class_label)
#print(information_gain(data[0], class_label))
#print(information_gain(data[2], class_label))