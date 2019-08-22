from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv
import graphviz


# 读取数据
Dtree = open('AllElectronics.csv','r')
reader = csv.reader(Dtree)
# 获取第一行数据
headers = reader.__next__()
print(headers)

# 对读取的数据进行处理
featureList = []
labelList = []

for row in reader :
    labelList.append(row[-1])
    # 建立数据字典(键值对)
    rowDict = {}
    # 对除去序号的属性建立键值对的关系
    for i in range(1,len(row)-1) :
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)

# 对每一行的数据转化为01表示，方便决策树进行判断
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
# 打印转化后的数据表示
print(x_data)
print(vec.get_feature_names())
print("labelList:"+str(labelList))
# 对标签数据进行转化
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_date"+str(y_data))

# 建立决策树模型 criterion='entropy'表示使用ID.3决策树模型
model = tree.DecisionTreeClassifier(criterion='entropy')
# 输入数据建立模型
model.fit(x_data, y_data)

# 测试
x_test = x_data[0]
predict = model.predict(x_test.reshape(1,-1))
print("predict :"+str(predict))

# 导出决策树
dot_data = tree.export_graphviz(model,
                                out_file = None,
                                feature_names = vec.get_feature_names(),
                                class_names = lb.classes_,
                                filled = True,
                                rounded = True,
                                )
print(lb.classes_)
graph = graphviz.Source(dot_data)
# 在同级目录下生成computer.pdf文件
graph.render('computer')
graph.view()

