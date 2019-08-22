import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("job.csv",delimiter=",")
x_data = data[1:,1]
y_data = data[1:,2]
# 修改数据格式
x_data = x_data[:,np.newaxis]
y_data = y_data[:,np.newaxis]

# 创建模型
linear_model = LinearRegression()
linear_model.fit(x_data,y_data)

# 定义多项式回归
# a+bx+cx**2
poly_features = PolynomialFeatures(degree=4)
x_poly = poly_features.fit_transform(x_data)
# 定义多项式回归模型（本质还是一个线性模型，对一元的数据X,进行了人为的扩维度）
poly_linear = LinearRegression()
poly_linear.fit(x_poly,y_data)

# 画图
plt.scatter(x_data,y_data,c="b")
plt.plot(x_data,poly_linear.predict(x_poly),c='r')
plt.plot(x_data,linear_model.predict(x_data),c='r')

plt.show()
