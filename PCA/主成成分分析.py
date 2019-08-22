import numpy as np
import matplotlib.pyplot as plt
# 数据的载入
data = np.genfromtxt("data.csv",delimiter=",")
x_data = data[:,0]
y_data = data[:,1]


# 数据的中心化
def Data_Centralization(dataMat) :
    maenData = np.mean(dataMat,axis=0)
    newData = dataMat-maenData
    return newData,maenData

# 求协方差矩阵
newdata,meandata = Data_Centralization(data)
covMat = np.cov(newdata,rowvar=0)

#求协方差矩阵的特征特征向量
eigenValue,eigenVector = np.linalg.eig(covMat)

# 将原始n维数据降成m维数据,,则选取欠m个最大的特征值对应的特征向量即可，再用原始数据与这些特征向量组成的矩阵相乘即可
eigenValueSortIndices = np.argsort(eigenValue)

# 这里将最大的特征值对应的特征向量，将数据降为一维的数据
top = 1
eigenValueTopIndices = eigenValueSortIndices[-1:(-1-top):-1]

# 找对应的特征向量
eigenVectorTop = eigenVector[:,eigenValueTopIndices]

# 降维
lowDimensiionData = np.mat(newdata)*np.mat(eigenVectorTop)

print(lowDimensiionData)










