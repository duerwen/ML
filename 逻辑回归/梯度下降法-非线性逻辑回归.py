import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
# 数据是否需要标准化
scale = False

data = np.genfromtxt("LR-testSet2.txt", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1, np.newaxis]


def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    # 切分不同类别的数据
    for i in range(len(x_data)):
        if y_data[i]==0:
            x0.append(x_data[i,0])
            y0.append(x_data[i,1])
        else:
            x1.append(x_data[i,0])
            y1.append(x_data[i,1])

    # 画图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    #画图例
    plt.legend(handles=[scatter0,scatter1],labels=['label0','label1'],loc='best')

def sigmod(x) :
    return 1/(1+np.exp(-x))

def loss(x_data,y_data,ws):

    xMatrix = np.mat(x_data)
    yMatrix = np.mat(y_data)
    ws = np.mat(ws)

    left = np.multiply(yMatrix,np.log(sigmod(xMatrix*ws)))
    right = np.multiply(1-yMatrix,np.log(sigmod(1-xMatrix*ws)))

    return -np.sum((left+right))/len(x_data)

def granDescent(x_data,y_data) :
    xMatrix = np.mat(x_data)
    yMatrix = np.mat(y_data)

    # 设置学习率，迭代次数，以及参数初始化等等
    lr = 0.03
    epochs = 50000
    ws = np.ones((xMatrix.shape[1],1))
    costList = []
    for i in range(epochs+1) :
        detal = (xMatrix.T*(sigmod(xMatrix*ws)-yMatrix))/xMatrix.shape[0]
        ws = ws - lr*detal
        if i%50 == 0 :
            costList.append(loss(x_data,y_data,ws))
    return ws,costList

# 对给定的数据进行处理(原始数据就变成非线性的了)
poly_fea = PolynomialFeatures(degree=3)
X_data = poly_fea.fit_transform(x_data)

ws,costList = granDescent(X_data,y_data)
plot()
plt.show()

# 画出等高面
# 获取数据值所在的范围
x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

z = sigmod(poly_fea.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(ws))# ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
for i in range(len(z)):
    if z[i] > 0.5:
        z[i] = 1
    else:
        z[i] = 0

# z = z.reshape(xx.shape) 相当于每个点都有一个对应的高度
z = z.reshape(yy.shape)

# 等高线图
cs = plt.contourf(xx, yy, z)
plot()
plt.show()

# 计算准确率与召回率
def predict(x_data,ws) :
    xMatrix = np.mat(x_data)
    ws = np.mat(ws)
    predictions = []
    for x in sigmod(xMatrix*ws):
        if x >= 0.5 :
            predictions.append(1)
        else :
            predictions.append(0)
    return predictions
predictions = predict(X_data,ws)
print(predictions)
print(classification_report(data[:, -1],predict(X_data,ws)))









