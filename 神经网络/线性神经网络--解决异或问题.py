#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 增加一些非线性的输入，x0(恒为1，对应的权值是神经元的偏置值),x1.x2,x1**2,x1*x2,x2**2

# 输入数据
X = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1]])
# 标签
Y = np.array([[-1],
              [1],
              [1],
              [-1]])

# 权值初始化，6行1列，取值范围-1到1
W = (np.random.random([6, 1])-0.5)*2

# 学习率设置
lr = 0.11
# 神经网络输出
O = 0

# 权值的更新函数，这里还是用的单层感知器中权值更新的公式
def update():
    global X, Y, W, lr
    # 这个地方与单层感知机不一样，这里的激活函数实际上就是y=x,线性神经网络的的激活函数是一个线性函数
    O = np.dot(X, W)
    W_C = lr*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C

for _ in range(10000):
    update()#更新权值

# 正样本
x1 = [0,1]
y1 = [1,0]
# 负样本
x2 = [0,1]
y2 = [0,1]

def calculate(x, root):
    a = W[5]
    b = W[2] + x*W[4]
    c = W[0] + W[1]*x +W[3]*x*x
    if root == 1 :
        return (-b+np.sqrt(b*b-4*a*c))/(2*a)
    if root == 2 :
         return (-b-np.sqrt(b*b-4*a*c))/(2*a)


xdata = np.linspace(-1,2)

# 画图，画出非线性超平面
plt.figure()

plt.plot(xdata,calculate(xdata,1),'r')
plt.plot(xdata,calculate(xdata,2),'y')

plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()

print(np.dot(X,W))





