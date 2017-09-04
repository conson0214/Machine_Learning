# -*- coding: utf-8 -*-
import svmMLiA
import matplotlib.pyplot as plt
from numpy import *

k1 = 1.3
dataArr,labelArr = svmMLiA.loadDataSet('testSetRBF.txt')
b,alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
svInd=nonzero(alphas.A>0)[0]
sVs=datMat[svInd] #get matrix of only support vectors
labelSV = labelMat[svInd];
print "there are %d Support Vectors" % shape(sVs)[0]
m,n = shape(datMat)
errorCount = 0
for i in range(m):
    kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],('rbf', k1))
    predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
    if sign(predict)!=sign(labelArr[i]): errorCount += 1
print "the training error rate is: %f" % (float(errorCount)/m)

support_vec_pos = []
support_vec_neg = []
for i in range(100):
    if alphas[i]>0.0:
        if labelArr[i] == 1:
            support_vec_pos.append(dataArr[i])
        elif labelArr[i] == -1:
            support_vec_neg.append(dataArr[i])

# trainning dataset
data_train = array(dataArr)
label_train = array(labelArr)
svec_pos  = array(support_vec_pos)
svec_neg  = array(support_vec_neg)
fig = plt.figure()
ax  = fig.add_subplot(111)
p1 = plt.scatter(data_train[label_train == 1, 0], data_train[label_train == 1, 1], marker='o', color='m', label='label+1', s=15)
p2 = plt.scatter(data_train[label_train == -1, 0], data_train[label_train == -1, 1], marker='o', color='c', label='label-1', s=30)
p3 = plt.scatter(svec_pos[:, 0], svec_pos[:, 1], marker='o', color='y', s = 120, facecolors = 'none')
p4 = plt.scatter(svec_neg[:, 0], svec_neg[:, 1], marker='o', color='r', s = 120, facecolors = 'none')
plt.title('data trainning')

dataArr,labelArr = svmMLiA.loadDataSet('testSetRBF2.txt')
errorCount = 0
datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
m,n = shape(datMat)
for i in range(m):
    kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],('rbf', k1))
    predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
    if sign(predict)!=sign(labelArr[i]): errorCount += 1
print "the test error rate is: %f" % (float(errorCount)/m)

# test dataset
data_test = array(dataArr)
label_test = array(labelArr)
fig = plt.figure()
ax  = fig.add_subplot(111)
p1 = plt.scatter(data_test[label_test == 1, 0], data_test[label_test == 1, 1], marker='o', color='m', label='label+1', s=15)
p2 = plt.scatter(data_test[label_test == -1, 0], data_test[label_test == -1, 1], marker='o', color='c', label='label-1', s=30)
plt.title('data test')
plt.show()


