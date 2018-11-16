import numpy as np
import copy
import random
import string
import graphviz as gv


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
    for row in range(0, len(attr_value)):
        for i in range(0, len(attr_value[row])):
            if attr_value[row][i] == "":
                attr_value[row][i] = random.randint(0, 1)
                
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
    labels = list(set(class_label))

    matrix = np.zeros((len(values), len(labels)))

    for i in range(len(attr_values)):
        matrix[values.index(attr_values[i]), labels.index(class_label[i])] += 1

    h_s = calcEntropy(matrix.sum(axis=0))
    h_s_a = 0

    for i in range(matrix.shape[0]):
        h_s_a += sum(matrix[i,:])/float(len(attr_values))*calcEntropy(matrix[i,:])
        #print(sum(matrix[i,:]),float(len(attrib)))

    #gain
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
        tree['gene'] = 'leaf'
        tree['children'] = []
        tree['value'] = [class_label.count(0), class_label.count(1)]
        tree['label'] = max(set(class_label), key = class_label.count)
        dims = data.shape
        tree['data samples'] = dims[1]
        return tree
    else:
        best_attribute = bestAttribute(data,class_label)
        
        tree['decision'] = best_attribute[0]
        tree['gain'] = best_attribute[1]
        tree['id'] = best_attribute[2]
        tree['gene'] = genes[best_attribute[2]]
        dims = data.shape
        tree['data samples'] = dims[1]
        tree['value'] = [class_label.count(0), class_label.count(1)]
        tree['children'] = []
        
        right_data,right_label,left_data,left_label = splitData(best_attribute,data,class_label)
        #returns one column!!
        tree['children'].append({})
        tdidt(genes,left_data,left_label,depth+1,tree['children'][-1])
        
        tree['children'].append({})
        tdidt(genes,right_data,right_label,depth+1,tree['children'][-1])

def tree_dot(outfile, tree):
    tree_dot = gv.Digraph(format='svg',engine='dot')
    traversal(tree, 'root', tree_dot)
    f = open(outfile,'w+')
    f.write(tree_dot.source)
    f.close() 
    
def traversal(current, position, tree_dot):
    if current['gene'] == 'leaf':
        tree_dot.attr('node', shape='box')
        name = position + str(random.choice(string.ascii_lowercase + string.digits))
        tree_dot.node(name, ''' samples = %(samples)d \n healthy = %(healthy)d, trisomic = %(trisomic)d \n class = %(class)d''' % {'samples': current['data samples'], 'healthy':current['value'][0], 'trisomic': current['value'][1], 'class':current['label']})
        tree_dot.edge(position, name)
    
    else:
        tree_dot.attr('node', shape='box')
        name = current['gene'] + '_' + str(current['data samples'])
        tree_dot.node(name = name, label = '''%(property_name)s >= %(dec)f \n samples = %(samples)d \n healthy = %(healthy)d, trisomic = %(trisomic)d''' % {'property_name': current['gene'], 'dec': current['decision'], 'samples':current['data samples'],'healthy':current['value'][0], 'trisomic': current['value'][1]})
        
        if position != 'root':
            tree_dot.edge(position, name)
    
    for children in current['children']:
        traversal(children, name, tree_dot)

def predict(tree, data):
    predicted_labels = []
    
    for row in data:
        predicted_labels.append(classify(tree, row))
    return predicted_labels

def classify(tree, data_row):
    tree_copy = tree
    while True:
        if tree_copy['gene'] == 'leaf':
            return tree_copy['label']
        elif data_row[tree_copy['id']] > tree_copy['decision']:
            tree_copy = tree_copy['children'][1]
            continue
        else:
            tree_copy = tree_copy['children'][0]
        
def accuracy(pred, true):
    correct = 0
    
    for pr, tr in zip(pred, true):
        if pr == float(tr):
            correct = correct + 1
    return float(correct)/len(pred)*100
    
    
print('Reading Data')
genes,all_data,class_label = readData("gene_expression_with_missing_values.csv")

print('Building tree')
tree = {}
tdidt(genes,all_data,[float(i) for i in class_label],0,tree)
tree_dot('tree4a.dot', tree)
print(tree)
print('Loading Testing Data')

file = [line.strip().split(',') for line in open('gene_expression_test.csv','r')]
genes = file[0][:-1]

test_data = [[float(el) for el in row[:-1]] for row in file[1:]]
test_label = [float(row[-1]) for row in file[1:]]
print('Testing Tree')
predicted = predict(tree, test_data)
accuracy = accuracy(predicted, test_label)
print('Decision Tree was ' + str(accuracy) + ' accurate')
