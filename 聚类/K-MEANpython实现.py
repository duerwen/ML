import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("kmeans.txt",delimiter=" ")

# 随机生成k个质心
def initCentroids(data,k) :
    # 初始化质心
    centroids = np.zeros((k,data.shape[1]))
    for i in range(k) :
        index = np.random.randint(0,data.shape[0])
        centroids[i,:] = data[index,:]
    #返回质心
    return centroids

def kmeans(data,k):
    # 是否改变质心
    changeCentroids = True
    # 样本数量与维度
    numSamples,dim = data.shape
    # 保存每个样本所属的类别与离质心的最短距离
    clusterData = np.zeros((numSamples,dim))
    # 初始化质心
    centroids = initCentroids(data,k)
    while changeCentroids :
        changeCentroids = False
        #计算每个样本离离质心的距离
        for i in range(numSamples) :
            # 初始化每个样本为O类别，且离质心的最短距离为minDistance = 0
            minDistance = 1000000
            label = 0
            # 计算第i个样本所属类别与离所属类别质心的距离
            for j in range(k) :
                # 计算第i个样本离第j个质心的距离
                distance = np.sqrt(np.sum((data[i,:]-centroids[j,:])**2))
                if distance < minDistance :
                    minDistance = distance
                    clusterData[i,1] = minDistance
                    label = j

            # 如果样本所属的类别发生变化，则需要更新质心(只要有一个样本的所属类别发生变化就更新质心)
            if clusterData[i,0] != label :
                clusterData[i,0] = label
                changeCentroids = True
        # 更新质心
        for j in range(k) :
            index = np.nonzero(clusterData[:,0] == j)
            # 计算第j类的质心
            centroids[j,:] = np.mean(data[index],axis=0)
    return centroids,clusterData

def showCluster(data,k,centroids,clusterData) :
    numSample,dim = data.shape
    maker = ['or', 'ob', 'oy', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    #遍历每个数据,花样本点
    for i in range(numSample) :
        plt.plot(data[i,0],data[i,1],maker[int(clusterData[i,0])])
    #画质心
    maker = ['*r', '*b', '*y', '*k', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(centroids.shape[0]) :
        plt.plot(centroids[i,0],centroids[i,1],maker[i],markersize=20)

    plt.show()


centroids,clusterData = kmeans(data,4)
print(centroids)
showCluster(data,4,centroids,clusterData)

