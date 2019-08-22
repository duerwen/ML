import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import linear_model


data = np.genfromtxt("LR-testSet.csv", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]


def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    # 切分不同类别的数据
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])
    # 画图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    # 画图例
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')

logisticModel = linear_model.LogisticRegression(solver="liblinear")
logisticModel.fit(x_data,y_data)

print(logisticModel.intercept_)
print(logisticModel.coef_[0][0])

#画决策边界
x = np.linspace(-4,3)
y =  x * (-logisticModel.coef_[0][0]/logisticModel.coef_[0][1]) - logisticModel.intercept_ / logisticModel.coef_[0][1]
plot()
plt.plot(x,y)
plt.show()

preditions = logisticModel.predict(x_data)
print(classification_report(y_data,preditions))