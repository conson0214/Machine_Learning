# -*- coding: utf-8 -*-
import svmMLiA
import matplotlib.pyplot as plt
from numpy import *

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
b,alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
w = svmMLiA.calcWs(alphas,dataArr,labelArr)

support_vec_pos = []
support_vec_neg = []
for i in range(100):
    if alphas[i]>0.0:
        r = labelArr[i]*(w[0]*dataArr[i][0]+w[1]*dataArr[i][1]+b)
        print dataArr[i],labelArr[i],alphas[i], r*labelArr[i]
        if labelArr[i] == 1 and abs(r - 1.0) < 0.0001:
        # if labelArr[i] == 1:
            support_vec_pos.append(dataArr[i])
        elif labelArr[i] == -1 and abs(r - 1.0) < 0.0001:
        # elif labelArr[i] == -1:
            support_vec_neg.append(dataArr[i])

data  = array(dataArr)
label = array(labelArr)
svec_pos  = array(support_vec_pos)
svec_neg  = array(support_vec_neg)

a  = float64(-w[0] / w[1])
xx = linspace(-2, 15)
#linspace取得-5到5之间的值用于画线
yy = a * xx - float64(b / w[1])
#点斜式方程这样就取得了yy的值
#后面就利用xx, 和yy的值画出分界的直线

b = array(support_vec_pos[0])
yy_down = a * xx + (b[1] - a * b[0])
b = array(support_vec_neg[0])
yy_up   = a * xx + (b[1] - a * b[0])
#得到分界线上方和下方与之平行的边际直线的xx和yy后面一并画出

fig = plt.figure()
ax  = fig.add_subplot(111)

p1 = plt.scatter(data[label == 1, 0], data[label == 1, 1], marker='o', color='m', label='label+1', s=15)
p2 = plt.scatter(data[label == -1, 0], data[label == -1, 1], marker='o', color='c', label='label-1', s=30)
p3 = plt.scatter(svec_pos[:, 0], svec_pos[:, 1], marker='o', color='y', s = 120, facecolors = 'none')
p4 = plt.scatter(svec_neg[:, 0], svec_neg[:, 1], marker='o', color='r', s = 120, facecolors = 'none')
plt.legend(loc='upper right')
plt.ylabel('x1')
plt.xlabel('x0')

plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.plot(xx, yy, 'k-')
#用plot画出3条线，第三个参数设置实线和虚线

plt.show()