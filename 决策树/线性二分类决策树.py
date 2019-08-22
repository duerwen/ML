import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz
data = np.genfromtxt("LR-testSet.csv",delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1]

plt.figure()
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
plt.show()

# 建立决策树模型
model = tree.DecisionTreeClassifier()
model.fit(x_data,y_data)

# 导出模型
# 得到决策树的文本化表示形式
dotdata = tree.export_graphviz(model,
                     out_file=None,
                     feature_names=['x','y'],
                     class_names=['label0','label1'],
                     filled=True,
                     rounded='True')
graph = graphviz.Source(dotdata)
graph.render("线性而分类决策树")
graph.view()

predictions = model.predict(x_data)
print(classification_report(predictions,y_data))