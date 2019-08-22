
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import operator
import random
import numpy as np

def knn(x_test,x_data,y_data,k) :
    # 计算样本数量
    x_data_size = x_data.shape[0]
    # 计算单个测试样本数据与所有测试数据的差值，因此对于每个样本数据都需要复制n份
    # n为测试数据的个数

    # (x_data_size,1)表示行复制六份，列保持不变，即原来有多少列现在还是多少列
    # 计算差值
    diffMat = np.tile(x_test,(x_data_size,1)) - x_data
    # 计算平方和
    sqDiffM = diffMat**2
    # 计算单个测试样本到每个训练数据的距离的平方和
    sqDistance = sqDiffM.sum(axis = 1)
    # 计算计算单个测试样本到每个训练数据的距离
    distance = sqDistance**0.5
    # 对这一组距离进行排序
    sortedDistance = distance.argsort()
    # 列表中是键值对
    classCount={}
    # 这个循环可以统计最邻近的K个样本点中各个类别的数量
    for i in range(k) :
        # 获取当前距离测试样本点距离最短的样本标签
        label = y_data[sortedDistance[i]]
        # 统计标签数量
        classCount[label] = classCount.get(label,0) + 1
    # 对样本标签的值进行统计进行排序，默认是从小到大排序，因此要取逆序，以键值对的value值进行排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

iris = datasets.load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.2) #分割数据0.2为测试数据，0.8为训练数据
predictions = []
for i in range(x_test.shape[0]):
    predictions.append(knn(x_test[i], x_train, y_train, 5))

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))