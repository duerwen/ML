from sklearn import tree
import numpy as np
import graphviz
data = np.genfromtxt("cart.csv",delimiter=",")
print(data)
x_data = data[1:,1:-1]
y_data = data[1:,-1]

# 建立模型
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(x_data,y_data)

# 到处决策树模型
dot_data = tree.export_graphviz(model,
                                out_file = None,
                                feature_names = ['house_yes','house_no','single','married','divorced','income'],
                                class_names = ['no','yes'],
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render("cart")
graph.view()