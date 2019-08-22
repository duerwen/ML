import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("data.csv",delimiter=",")
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
# 给x_data增加一列偏置值1,axis=1表示在列的方向进行合并
X_data = np.concatenate((np.ones((100,1)),x_data),axis=1)
# 标准方程法求解权值矩阵
def weights(x_data,y_data) :
    xMatrix = np.mat(x_data)
    yMatrix = np.mat(y_data)
    xTx = xMatrix.T*xMatrix
    if np.linalg.det(xTx) == 0 :
        print("this martrix can not inverse")
        return
    ws = xTx.I*xMatrix.T*yMatrix
    return ws

ws = weights(X_data,y_data)
print(ws[0,0])
# 画图
plt.scatter(x_data,y_data,c="r")
plt.plot(x_data,ws[0,0]+ws[1,0]*x_data,c="b")
plt.show()

