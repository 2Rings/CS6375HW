
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import mglearn
from sklearn.svm import LinearSVC
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from numpy import linalg as LA

matrix=np.loadtxt('mystery.data',delimiter=',')
matrix=np.hsplit(matrix,np.array([4]))
X=matrix[0]
y=matrix[1]
train_num=1000
feature=4
XX = np.zeros((train_num,34))
def phi():
    for i in range(0,train_num):
        f=0
        for j in range(0,feature):
            XX[i][f]= X[i][j]
            f+=1
        for k in range(0, feature):
            for p in range(k,feature):
                XX[i][f]=X[i][k]*X[i][p]
                f+=1
        for m in range(0,feature):
            for t in range(m, feature):
                for r in range(t,feature):
                    XX[i][f]=X[i][m]*X[i][t]*X[i][r]
                    f+=1

with open('a', 'w') as f:
    for row in XX:
        line=""
        for e in row:
            line+= str(e)+' '
        f.write(line+'\n')

def margin(sup):
    for i in range(0,feature):
        maxMargin=np.fabs(np.dot(weight, sup)+b)/LA.norm(weight,2)
    return maxMargin

def fx():
    err_count=0
    for i in range(0,train_num):
        sum =y[i]*(np.dot(weight,XX[i])+b)
        if sum < 0:
            err_count+=1
    return err_count

phi()


#X, y = make_blobs(centers=4, random_state=8)
#y = y % 2
# preprocessing using 0-1 scaling
'''
def final_w():
    weight=np.zeros((1,4))[0]
    for i in range(0,4):
        for k in range(0,1000):
            weight[i]+=alpha[k]*X[k][i]*y[k]
    return weight
def predict(w,b,offset,test_num):
    err=0
    for i in range(offset,test_num+offset):
        fy=np.dot(w,X[i])+b
        if  (fy >= 0 and Y[i] <0) or (fy < 0 and Y[i] > 0):
            err +=1
    return float(err)/test_num
'''
scaler = MinMaxScaler()
scaler.fit(XX)
X_train_scaled = scaler.transform(XX)
X_test_scaled = scaler.transform(XX)

linear_svm = SVC(kernel='linear',degree=1,C=2000.0,gamma=0.1).fit(X_train_scaled, y)
alpha = linear_svm.decision_function(X_train_scaled)
supVec=linear_svm.support_vectors_
weight=linear_svm.coef_
b=linear_svm.intercept_
maxMargin1=margin(supVec[0])
maxMargin2=margin(supVec[1])
#w=final_w()
err_count=fx()
print linear_svm
print "alpha:", linear_svm.dual_coef_
print "correct: ",(linear_svm.score(X_train_scaled, y))
print "err_count:", err_count
print "Support Vector:", linear_svm.support_vectors_
print "weight:", linear_svm.coef_
print "b:", linear_svm.intercept_
print "maxMargin:", maxMargin1

#print linear_svm.decision_function(X_train_scaled)
#print linear_svm.predict(X_train_scaled)
#print w
'''
mglearn.plots.plot_2d_separator(linear_svm, X)
plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
plt.xlabel("feature1")
plt.ylabel("feature2")

# add the squared first feature
X_new = np.hstack([X, X[:, 1:] ** 2])
figure = plt.figure()
# visualize in 3D
ax = Axes3D(figure, elev=-152, azim=-26)
ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=y, cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature1")
ax.set_ylabel("feature2")
ax.set_zlabel("feature1 ** 2")
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# show linear decision boundary
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min(), X_new[:, 0].max(), 50)
yy = np.linspace(X_new[:, 1].min(), X_new[:, 1].max(), 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=y, cmap=mglearn.cm2, s=60)
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.set_xlabel("feature1")
ax.set_ylabel("feature2")
ax.set_zlabel("feature1 ** 2")
plt.show()
'''
