import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
'''
要事先安装pydotplus和graphviz, pydotplus就用pip来安装即可
但是graphviz需要去官网下载.msi安装包进行安装，然后将bin目录添加到系统环境变量中
'''
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 在可视化一根树的时候，export_graphviz这个方法很强大
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("iris.pdf") 
graph.write_png("iris.png")
