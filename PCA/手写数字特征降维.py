from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix

#PCA原理分析：https://www.jianshu.com/p/6662849038e5

digits = load_digits()
x_data = digits.data
y_data = digits.target


# top:目标维度
def PCA(dataMat,top) :
    # 样本中心化
    meanData = np.mean(dataMat,axis=0)
    newData = dataMat-meanData
    # 求协方差矩阵(每一行是一个数据)，各个维度之间的协方差矩阵
    covMat = np.cov(newData,rowvar=0)
    # 求协方差矩阵的特征值特征向量
    eigenValue,eigenVector = np.linalg.eig(covMat)
    #对协方差矩阵的特征值排序
    eigenValueIndices = np.argsort(eigenValue)
    # 选区前top个最大的特征值
    eigenValueIndicesTop = eigenValueIndices[-1:(-1-top):-1]
    # 选择对应的特征向量
    eigenVectorTop = eigenVector[:,eigenValueIndicesTop]
    # 数据的降维
    lowDimensionData = np.mat(dataMat)*np.mat(eigenVectorTop)
    return lowDimensionData

# 将原始数据降低为二维
X_data = PCA(x_data,20)
x_train,x_test,y_train,y_test = train_test_split(X_data,y_data) #分割数据1/4为测试数据，3/4为训练数据

# 利用神经网络训练
model = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500)
model.fit(x_train,y_train)


# 画出测试急的分类图
# x = np.array(x_test)[:,0]
# y = np.array(x_test)[:,1]
# z = np.array(x_test)[:,2]
# plt.scatter(x,y,z,c=model.predict(x_test))
# plt.show()

# 看看分类的正确性
print(classification_report(y_test,model.predict(x_test)))