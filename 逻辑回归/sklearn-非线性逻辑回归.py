import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import PolynomialFeatures

x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2,n_classes=2)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()

# 模型训练
logistic = linear_model.LogisticRegression(solver="liblinear")
poly_fea = PolynomialFeatures(degree=3)
x_poly = poly_fea.fit_transform(x_data)
logistic.fit(x_poly, y_data)
print(logistic.score(x_poly,y_data))
# 画图
x_min,x_max = x_data[:,0].min()-1,x_data[:,0].max()+1
y_min,y_max = x_data[:,1].min()-1,x_data[:,1].max()+1

# 画出网格矩阵
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
# 对网格矩阵中每一个点都求对应的预测值
z = logistic.predict(poly_fea.fit_transform(np.c_[xx.ravel(),yy.ravel()]))
z = z.reshape(xx.shape)
# 画出等高线图
plt.contourf(xx,yy,z)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()


