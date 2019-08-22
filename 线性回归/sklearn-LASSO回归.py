import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

data = genfromtxt(r"longley.csv",delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1]


# 创建模型
model = linear_model.LassoCV()
model.fit(x_data,y_data)
# LASSO回归的alpha
print(model.alpha_)
print(model.coef_)