import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data = np.genfromtxt("longley.csv",delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1]

# 创建模型

# 默认是50个值
alphas = np.linspace(0.001,1)
# 使用岭回归(本质是一个使用了L2正则化的线性回归)做模型，CV表示cross validation表示交叉验证
modle = linear_model.RidgeCV(alphas=alphas,store_cv_values=True)
modle.fit(x_data,y_data)

# 岭回归系数，对于50个岭回归alpha系数，会在每一次的交叉验证中产生一个loss值，对这些loss值取平均数就是该岭回归alpha系数
# 对应的loss值，对所有的岭回归alpha系数用一样的方法求得对应的loss值，再取最小的loss值对应的alpha系数就是最后的alpha系数
print(modle.alpha_)
# loss值,打印；loss的shape
print(modle.cv_values_.shape)

# 画图
plt.plot(alphas,modle.cv_values_.mean(axis=0))
plt.scatter(modle.alpha_,min(modle.cv_values_.mean(axis=0)),c="r")
plt.show()

# 做预测
print(modle.predict(x_data[2,np.newaxis]))
