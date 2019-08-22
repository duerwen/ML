import numpy as np
import matplotlib.pyplot as plt
# 数据的切分
data = np.genfromtxt("longley.csv",delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1,np.newaxis]
# 给数据加上偏置项
X_data = np.concatenate((np.ones((16,1)),x_data),axis=1)
# 利用标准方程法求岭回归的特征参数,这个lam其实就是L2正则化的系数
def weights(x_data,y_data,lam=0.2) :
    xMatrix = np.mat(x_data)
    yMartrix = np.mat(y_data)
    # 这里这点与标准方程发求截线性回归不一样,np.eye()生成一个对角矩阵
    rxTx = xMatrix.T*xMatrix+np.eye(x_data.shape[1])*lam

    if np.linalg.det(rxTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    ws = rxTx.I * xMatrix.T * yMartrix
    return ws

ws = weights(X_data,y_data)
print(ws)


# 计算预测值
print(np.mat(X_data)*np.mat(ws))