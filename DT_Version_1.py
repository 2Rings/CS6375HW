'''
Description: Decision_Tree_ID3
@author: huilin
@version: 2
@date: 09/20/2017

'''

import numpy as np
import re
from collections import defaultdict,namedtuple
import uuid
import matplotlib.pyplot as plt
from graphviz import Digraph
# https://docs.python.org/2/library/re.html


pattern = r'=([a-z?])'
pmatch = re.compile(pattern)
attr_Info=['bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s',
            'fibrous=f,grooves=g,scaly=y,smooth=s',
            'brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y',
            'bruises=t,no=f',
            'almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s',
            'attached=a,descending=d,free=f,notched=n',
            'close=c,crowded=w,distant=d',
            'broad=b,narrow=n',
            'black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y',
            'enlarging=e,tapering=t',
            'bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?',
            'fibrous=f,scaly=y,silky=k,smooth=s',
            'fibrous=f,scaly=y,silky=k,smooth=s',
            'brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y',
            'brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y',
            'partial=p,universal=u',
            'brown=n,orange=o,white=w,yellow=y',
            'none=n,one=o,two=t',
            'cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z',
            'black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y',
            'abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y',
            'grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d']

    #we need to know the values of each featrues
    #attr={key: i, values}, i is number
y_label=[]
attr_List={}
for i in range(len(attr_Info)):
    m=attr_Info[i]
    #findall the information match the pattern,
    # and return a list
    values= pmatch.findall(m)
    attr_List[i]=values
    #return attr, which has lists of features

#class Decision_Tree_ID3(object):

    # Entropy:
    def Entropy(y):
        H=0.0
        prob=0.0
        dictLab=label(y)
        for lab in dictLab:
            prob= float(len(lab))/float(len(y))
            H-= prob*np.log2(prob)
        return H

    #calculate size(y_i)
    def label(y):
        dictLab={}
        for lab in y:
            if label in dictLab:
                dictLab[lab]+=1
            else:
                dictLab[lab]=1
        return dictLab

    def attr_v(X,y,v,attr_th):
        y_sub=[]
        for i in range(len(X)):
            if(X[i][attr_th]==v):
                y_sub.append(y[i])
        return y_sub
    #inforGain helps to find which attr provide largest inforGain
    def inforGain(X,y,attr,attr_L):
        #Entropy
        Gain=Entropy(y)
        # //attrList=attr_List(attr_Info
        values=attr_List[attr]
        attr_th=attr_L.index(attr)
        for v in values:
            #calculate how many data with v
            y_v = attr_v(X,y,v,attr_th)
            # D_i=len(y_v),D = len(y), y is all data
            #Entropy(y_v) calculates C_ki/D_i
            #Gain= Entropy-Conditional Entropy
            Gain=Gain-float(len(y_v))/float(len(y))*Entropy(y_v)
        return Gain #type Double/float

    #inforGain: choose the best featrues and return its index
    def select_attr(X,y,attr_L):
        attr_Gain=[]

        for attr in attr_L:
            #get all Gain of each attr in the attr_L, store them in list
            attr_Gain.append(inforGain(X,y,attr,attr_L))
        #print "attr_Gain: ", attr_Gain
        #find the maxGain and return its index
        #print "attr_L: ", attr_L, "idx: ", attr_Gain.index(max(attr_Gain))
        return  attr_Gain.index(max(attr_Gain))#type Int

    def split_into_subSet(X,y,attr_idx):
        split_dict={}
        #split_dict has this form: key: attr_Val, value: (X_sub,y_sub)
        for x, label in zip(X,y):
            attr_Val= x[attr_idx]
            sub_X, sub_Label=split_dict.setdefault(attr_Val,[[],[]])
            sub_X.append(x[:attr_idx]+x[attr_idx+1:])
            sub_Label.append(label)
        #print "sub_Label: ", sub_Label
        return split_dict,sub_Label # dict{}

    #vote
    def get_Majority(y):
        label_num=defaultdict(lambda:0)# if null assign 0
        for label in y:
            label_num[label]+=1
        return max(label_num,key=label_num.get)# return Label, type string

    def cut(y):
        percent=0.0
        label_num=defaultdict(lambda:0)# if null assign 0
        for label in y:
            label_num[label]=label_num[label]+1
        percent=label_num[max(label_num,key=label_num.get)]/float(len(y))
        return percent# return Label, type string

    def create_Tree(X, y, attr_L):

        if len(set(y))==1:
            return y[0]
        #all featrues traversed
        if len(attr_L)==0:
            return get_Majority(y)#return string: label

        #get the best attr for a node
        node_attr_idx = select_attr(X,y,attr_L)

        attr = attr_L[node_attr_idx]
        #print "node_attr_idx: ",node_attr_idx, "attr: ", attr
        #create a subTree under node(node_attr_idx)


        #copy the attr_L
        sub_attr_L=attr_L[:]
        #pop the attr_idx that have been used
        sub_attr_L.pop(node_attr_idx)
        #get subSet after poping node_attr_idx
        split_dict,sub_Label=split_into_subSet(X,y,node_attr_idx)

        #if len(set(sub_Label))==1:
        #    return sub_Label[0]
        #create_subTree
        tree={}
        tree[attr]={}
        for attr_val, (X_sub,y_sub) in split_dict.items():
            #recursive
            tree[attr][attr_val] = create_Tree(X_sub,y_sub,sub_attr_L)
            #self.tree=tree
            #self.attr_L=attr_L
            #print attr_L

        return tree #return tree{,{}}



    def traverse(x,tree):
        #select featrues
        if isinstance(tree,str):
            y_label.append(tree)
        else:
            attr_type=tree.keys()[0]#Node
            #print "attr_type: ", attr_type
            #get featrues sub_tree
            attr_tree=tree.values()[0]#Edges
            #print "attr_tree: ", attr_tree
            #choose the corresponding subtree
            #if x[attr_type] not in atrr_tree:
            #print "attr_tree[x[attr_type]]: ", attr_tree[x[attr_type]]
            subtree=attr_tree[x[attr_type]]
            traverse(x,subtree)

    def predict_Tree(X_test,tree):
        for x in range(len(X_test)):
            traverse(X_test[x],tree)

    def test_Tree(y_test):
        error=0.0
        for i in range(len(y_test)):
            if(y_test[i]!=y_label[i]):
                error+=1
        return 1.0-float(error/len(y_test))

    def get_Nodes_Edges(tree=None,root_node=None):
        Node=namedtuple('Node',['id','label'])
        Edge = namedtuple('Edge',['start','end','label'])

        if tree is None:
            tree=tree

        if type(tree) is not dict:
            return [],[]

        nodes,edges=[],[]

        if root_node is None:
            label=list(tree.keys())[0]
            root_node=Node._make([uuid.uuid4(),label])
            nodes.append(root_node)

        for edge_label,sub_tree in tree[root_node.label].items():
            node_label=list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
            sub_node=Node._make([uuid.uuid4(),node_label])
            nodes.append(sub_node)

            edge=Edge._make([root_node,sub_node,edge_label])
            edges.append(edge)

            sub_nodes,sub_edges=get_Nodes_Edges(sub_tree,root_node=sub_node)
            #append more than one values at a time
            nodes.extend(sub_nodes)
            edges.extend(sub_edges)

        return nodes,edges

    def dotify(tree=None):
        #Graphviz
        #
        if tree is None:
            tree = tree

        content = 'digraph decision_tree {\n'
        Nodes, Edges = get_Nodes_Edges(tree)

        for node in Nodes:
            content += '    "{}" [label="{}"];\n'.format(node.id, node.label)

        for edge in Edges:
            start, label, end = edge.start, edge.label, edge.end
            content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label)
        content += '}'

        return content

    def parseFile(filename):
        # //attr_Info=attr_pattern()
        #//train_file=open(filename,'rU')
        #//Train_Data=np.loadtxt(train_file, dtype=str, delimiter=',')
        #//num,attribute=Train_Data.shape
        X,y=[],[]
        f=open(filename,'rU')
        for line in f.readlines():
                y.append(line.split(',')[0])
                X.append(line.split(',')[1:])
        attr_L=[i for i in range(len(attr_Info))]
        return attr_L, X, y

#warn: attr_L is a list, and has attr_idx and attr, they are different, and we want to use idx.
if __name__ == '__main__':
    #clf=Decision_Tree_ID3()
    attr_L,X,y=parseFile('mush_train.data')
    tree=create_Tree(X,y,attr_L)
    attr_test,X_test,y_test = parseFile('mush_test.data')
    predict_Tree(X_test,tree)
    test_accuracy=test_Tree(y_test)
    #print tree
    with open('mush.dot','w') as f:
        dot=dotify(tree)
        f.write(dot)

    #dotfile.render('dc.gv',view=True)
    #for i in range(len(X_test)):
    #    print "y_label: ", y_label[i], "vs y_test: ", y_test[i]
    print test_accuracy
