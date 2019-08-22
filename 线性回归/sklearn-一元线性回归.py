from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 数据的读取
data = np.genfromtxt("data.csv",delimiter=',')
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
print(x_data)
# 得到模型
model = LinearRegression()
model.fit(x_data, y_data)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
