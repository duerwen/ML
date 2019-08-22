import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
Standardization = True



data = np.genfromtxt("LR-testSet.csv",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]
# 给数据添加偏置项
X_data = np.concatenate((np.ones((x_data.shape[0],1)),x_data),axis=1)


def sigmod(x) :
    return 1/(1+np.exp(-x))

def lossFunction(xMat,yMat,ws) :

    left = np.multiply(yMat,np.log(sigmod(xMat*ws)))
    right = np.multiply((1-yMat),np.log(1-sigmod(xMat*ws)))

    loss = (-1/len(xMat))*np.sum(left+right)
    return loss

def granDescent(xMat,yMat) :
    #定义学习率，迭代次数,权值初始化等
    lr = 0.001
    epochs = 10000
    ws = np.mat(np.ones((xMat.shape[1],1)))
    costList = []
    if Standardization == True :
        xMat = preprocessing.scale(xMat)
    for i in range(10000) :
        #预测值
        h = sigmod(xMat*ws)
        ws_detal = (xMat.T*(h-yMat))/xMat.shape[0]
        ws = ws - lr*ws_detal

        if i%50 ==0 :
            costList.append(lossFunction(xMat,yMat,ws))

    return ws,costList


def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    for i in range(len(x_data)):
        if y_data[i] == 1:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])
        else:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])

    plt.scatter(x0, y0, c="r")
    plt.scatter(x1, y1, c="b")

def perdict(xMat,ws):
    perdictions = []
    if Standardization == True :
        xMat = preprocessing.scale(xMat)
    for x in sigmod(xMat*ws) :
         if x >= 0.5 :
             perdictions.append(1)
         else:
             perdictions.append(0)
    return perdictions

# 训练模型
xMat = np.mat(X_data)

yMat = np.mat(y_data[:,np.newaxis])
ws,costList = granDescent(xMat,yMat)
plot()
plt.show()

# 画出分类边界
if Standardization == False :
    x = [[-4],[3]]
    y = x * (-ws[1] / ws[2]) - ws[0] / ws[2]
    plt.plot(x, y)
    plt.show()

# 画出loss值的变化
x = np.linspace(0, 10000, 200)
plt.plot(x, costList)
plt.show()

perdictions = perdict(xMat,ws)
print(classification_report(y_data,perdictions))
print(perdictions)










