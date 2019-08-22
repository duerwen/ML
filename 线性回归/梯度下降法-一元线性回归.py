import numpy as np
import matplotlib.pyplot as plt

# 数据的处理
data = np.genfromtxt("data.csv",delimiter=",")
x_data = data[:,0]
y_data = data[:,1]

# 学习率
lr = 0.0001
# 参数
b = 0
k = 0
# 求解梯度的次数
epochs = 50

# 梯度下降算法涉及到学习率，参数，特征值，标记值
def gradient_descent(x_data,y_data,lr,b,k,epochs) :
    m = float(len(x_data))
    # 做很多次的迭代
    for j in range(epochs) :
        delta_b = 0
        delta_k = 0
        # 梯度下降（批量梯度下降算法）
        for i in range(0,len(x_data)) :
            delta_b = (((x_data[i]*k)+b)-y_data[i])*(1/m) + delta_b
            delta_k = (((x_data[i]*k)+b)-y_data[i])*x_data[i]*(1/m) + delta_k
        b = b - (lr*delta_b)
        k = k - (lr*delta_k)
    return b,k

# 最小二乘法计算损失
def lossfuntion(x_data,y_data,k,b) :
    m = float(len(x_data))
    loss = 0
    for i in range(0,len(x_data)) :
        loss = loss+((x_data[i]*k+b)-y_data[i])**2
    return loss*(1/2)*(1/m)

# 一元线性回归—梯度下降求解
print("starting b = {0} ,k={1} ,loss={2}".format(b,k,lossfuntion(x_data,y_data,k,b)))
print("running")
b,k = gradient_descent(x_data,y_data,lr,b,k,epochs)
print("starting b = {0} ,k={1} ,loss={2}".format(b,k,lossfuntion(x_data,y_data,k,b)))

plt.plot(x_data,x_data*k+b,c = 'r')
plt.scatter(x_data,y_data,c = 'b')
plt.show()

