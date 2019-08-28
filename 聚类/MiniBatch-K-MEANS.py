from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("kmeans.txt",delimiter=" ")

model = MiniBatchKMeans(n_clusters=4)
model.fit(data)
#打印质心点
print(model.cluster_centers_)
#输出预测值
print(model.predict(data))
print(model.predict(data[0,np.newaxis]))

#画图

x_min,x_max = data[:,0].min()-1,data[:,0].max()+1
y_min,y_max = data[:,1].min()-1,data[:,1].max()+1
# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
z = model.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
#画等高线图
plt.contourf(xx,yy,z)
#画数据点
maker =  ['or', 'ob', 'oy', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
for i in range(data.shape[0]) :
    index = int(model.predict(data[i,np.newaxis]))
    plt.plot(data[i,0],data[i,1],maker[index])
#画质心
maker = ['*r', '*b', '*y', '*k', '^r', '+r', 'sr', 'dr', '<r', 'pr']
centroids = model.cluster_centers_
for i in range(centroids.shape[0]) :
    plt.plot(centroids[i, 0], centroids[i, 1], maker[i], markersize=20)

plt.show()