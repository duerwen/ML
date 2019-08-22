#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
# 数据 单层感知机只能对数据进行线性划分，因此无法对异或问题的数据进行划分
X = np.array([[1,0,1],
              [1,1,2],
              [1,0,0],
              [1,2,0]])
# 标记
Y = np.array([[1],
             [1],
             [-1],
             [-1]])

# 权值
W = (np.random.random([3,1])-0.5)*2

# 学习率
lr = 0.11

# 神经网络的输出
O = 0

# 对权值进行更新
def update():
    global O,X,Y,W,lr
    O = np.sign(np.dot(X,W))
    W_C = lr*X.T.dot(Y-O)/int(X.shape[0])
    W = W + W_C
for i in range(1000) :
    update()
    O = np.sign(np.dot(X,W))
    print(O)
    if(O == Y).all() :
        break

# 画图
# 正样本
X1 = [0,1]
Y1 = [1,2]
# 负样本
X2 = [0,2]
Y2 = [0,0]

# 分界线的计算
k = -W[1]/W[2]
d = -W[0]/W[2]

# 图像的横坐标
xdata = np.arange(-2,5)

plt.figure()
plt.plot(xdata,k*xdata+d,'r')
plt.scatter(X1,Y1,c='b')
plt.scatter(X2,Y2,c='y')
plt.show()