import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = np.genfromtxt("Delivery.csv",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]

# 学习率
lr = 0.0001
# 所求参数，初值设置为0
theta0 = 0
theta1 = 0
theta2 = 0

# 最大迭代次数
epochs = 1000

# 梯度下降求解参数值
def gradient_descent_function(x_data,y_data,theta0,theta1,theta2,lr,epochs) :
    m = float(len(x_data))
    for i in range(epochs) :
        grad_theta0 = 0
        grad_theta1 = 0
        grad_theta2 = 0
        for j in range(len(x_data)) :
            grad_theta0 += ((theta0+theta1*x_data[j,0]+theta2*x_data[j,1])-y_data[j])*(1/m)
            grad_theta1 += ((theta0+theta1*x_data[j,0]+theta2*x_data[j,1])-y_data[j])*x_data[j,0]*(1/m)
            grad_theta2 += ((theta0+theta1*x_data[j,0]+theta2*x_data[j,1])-y_data[j])*x_data[j,1]*(1/m)
        theta0 = theta0 - lr*grad_theta0
        theta1 = theta1 - lr*grad_theta1
        theta2 = theta2 - lr*grad_theta2
    return theta0,theta1,theta2

def loss_function(x_data,y_data,theta0,theta1,theta2) :
    loss = 0
    for j in range(len(x_data)) :
        loss += ((theta0+theta1*x_data[j,0]+theta2*x_data[j,1])-y_data[j])**2
    return loss*(1/2)*float(len(x_data))

print("Starting theta0 = {0}, theta1 = {1}, theta2 = {2}, error = {3}".
      format(theta0, theta1, theta2,loss_function(x_data,y_data,theta0,theta1,theta2)))
print("Running...")
theta0, theta1, theta2 = gradient_descent_function(x_data,y_data,theta0,theta1,theta2,lr,epochs)
print("After {0} iterations theta0 = {1}, theta1 = {2}, theta2 = {3}, error = {4}".
      format(epochs, theta0, theta1, theta2, loss_function(x_data,y_data,theta0,theta1,theta2)))

# 画出三维图像

# 画出点在三维图像中的位置
ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o', s = 30) #点为红色三角形
# 画出分割平面
x0 = x_data[:,0]
x1 = x_data[:,1]
# x0,,x1对应位置就有确定这个平面的所有点
x0, x1 = np.meshgrid(x0, x1)
# 得到的z是一个二维矩阵，每个位置的值对应一个XOY平面上的点的高度
z = theta0 + theta1*x0 + theta2 * x1

ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
ax.plot_surface(x0, x1, z)

plt.show()
