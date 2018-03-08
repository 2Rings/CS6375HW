'''
Description: SVM
@author: huilin
@version: 2
@date: 09/09/2017

'''

from cvxopt import matrix
import numpy as np
from cvxopt.solvers import qp

def SVM(Train_Data):
        train_num,features =Train_Data.shape
    #extract the information of row and column, also X,y
        y = Train_Data[:,0]
        dimX= features -1
    #get label

    # Add one col for X
        X= Train_Data[:,1:features]
    # Feature mapping: phi(x)=(1,x,x**2,x**3)
        for i in range(dimX):
            for j in range(i,dimX):
                for k in range(j,dimX):
                    X_3= X[:,i]*X[:,j]*X[:,k]
                    X=np.hstack((X,X_3.reshape(train_num,1)))
    #the first column of X is a vector(1,1,...,1)
        X=np.hstack((Train_Data[:,1:features],X))

        #X=np.hstack((np.ones(train_num).reshape(train_num,1),X))
    #know how many feartures after mapping
        train_num, dimX=X.shape
    #Put bias(b) into w as the first element. Which means b=w_0.
    #Make sure that b actually don't exist in the dot(W.T,W)
    #In this case, P is a diag(1,...,1) matrix, dimension = dimX
        P=matrix(np.identity(dimX))#Construct a identity diag matrix
        print dimX
        P=matrix(np.hstack((P,np.zeros(dimX).reshape(dimX,1))))#add one zero column
        P=matrix(np.vstack((P,np.zeros(dimX+1).reshape(1,dimX+1))))#add one zero row ,P[0][0] is bias
        print P
        q=matrix(np.zeros(dimX+1))
        G=matrix(-y.reshape(train_num,1)*np.hstack((X,np.ones(train_num).reshape(train_num,1))))
        h=matrix(-1*np.ones(train_num).reshape(train_num,1))
    #linear programing
        weight_b = qp(P,q,G,h)

        weight=np.array(weight_b["x"][0:dimX])
        b=np.array(weight_b["x"][dimX])
    #calculate the cortectness
        miss = 0
        for i in range(train_num):
            xi = X[i]
    	    yi = np.dot(weight.T,xi)+b
    	    if(np.sign(yi) != np.sign(y[i])):
                miss = miss+1
    #return
        return weight,b,miss

def fx(w ,b, x):
	return np.dot(w,X)+b
'''
def check(w, b, X):
	miss = 0
    num,featrues=X.shape
	for data in X:
		X = data[1:featrues]
		y = data[0]
		yi = fx(w, b, X)
		if(np.sign(yi) != np.sign(y)):
			miss = miss+1
	return miss
    '''
#open a file, in read and Usual '\n'
file = open("wdbc_train.data","rU")
#Load the data in double and use "," to separate different feartures and lable
Train_Data = np.loadtxt(file,dtype=float, delimiter=",")

w,b,miss=SVM(Train_Data)

#miss=check(w,b,X)
print("w: ")
print w
print "b: "
print b
print "# miss:"
print miss
