#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
Created on Aug. 2017
Description：perceptron
@author: Hui LIN
@version: 2.4
'''
import sys
import os
import random

def get_xy(filename,data_num,offset):
    matrix = []
    f = open(filename,'r')
    lines = f.readlines()#get a list, each of its component is a line
    for i in range(offset,offset+data_num):
        matrix.append(lines[i].split(','))#add element
    f.close()
    return matrix
    
def fx(w,b,xy):
    res = 0
    for i in range(0,4):
            res += w[i]*xy[i]
    res += b 
    return res
 
def renew(w,b,xy,yt):
    for i in range(0,4):
            w[i] = w[i]+yt*xy[4]*xy[i]
    b+=yt*xy[4]
    return [w,b]
 
def perceptron_train(w,b,filename,train_num,maxiter):
    matrix = get_xy(filename,train_num,0)
    k = 0
    iters =0
    err=1
    yt=0.7
    while err!=0 and iters<maxiter:#maxIter:
        err=0
        for i in range(0,train_num):
            xy = [float(x) for x in matrix[i]]
            r = fx(w,b,xy)
            if (r >=0 and xy[4] <0) or (r <0 and xy[4] > 0):#equal：r*matrix[i][1]<0,misclassified
                w_b = renew(w,b,xy,yt)#iter:w<--w+yt*yi*xi,b<--b+yt*yi*xi
                w=w_b[0]
                b=w_b[1]
                err +=1 #error_count+1
        k+=err #error+1
        iters +=1 #iterator+1
        if(iters<4):
            print "iters:",iters
            print "w:", w
            print "b:", b
    return [w,b,k,iters]
 
def perceptron_test(w,b,filename,test_num,offset):
    matrix = get_xy(filename,test_num,offset)
    err=0
    for i in range(0,test_num):
        xy = [float(x) for x in matrix[i]]
        r = fx(w,b,xy)
        if  (r >= 0 and xy[4] <0) or (r < 0 and xy[4] > 0):
            err +=1
    return float(err)/test_num
 
if __name__ == '__main__':
    
    w = [0,0,0,0]
    b = 0
    model = perceptron_train(w,b,'perceptron.data',1000,200)
    print ("---  iterations : ---" %(model[3]))
    print "w:", model[0]
    print "b:", model[1]
    print ("perceptron test error:%f" %(perceptron_test(model[0],model[1],'perceptron.data',1000,0)))
