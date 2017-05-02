import numpy as np
import sklearn.tree as st
import pydotplus

'''
本节主要关于决策树是怎么处理连续数值的，最简单的策略是采用二分法来进行处理，这是C4.5中采用的机制。
给定样本集D和连续属性a，假定a在D上出现了n个不同的取值，将这些值从小到大排序{a1, a2, ...., an}, 然后根据相邻
两个数据ai, a(i+1)的均值，得出另一个数组{b1, b2, ..., b(n-1)}，每一个bi = (ai + a(i+1)) / 2, 然后逐个选取bi来
给连续属性a分类，小于bi的是一类，大于bi的是另一类，然后计算信息增益或者基尼指数，所以对于连续属性来说，每一个
都是二分类问题
'''
X_train = [[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343,\
            0.639, 0.657, 0.360, 0.593, 0.719],
           [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099,\
            0.161, 0.198, 0.370, 0.042, 0.103]]
X_labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

clf = st.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(np.transpose(X_train), np.transpose(X_labels))
dot_data = st.export_graphviz(clf, out_file=None, 
                         feature_names=['density', 'sugar prop'],  
                         class_names=['1', '0'],  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("watermelon.pdf") 
graph.write_png("watermelon.png")

# 结果中可以看到每个test node的阈值都是某两个相邻数值的均值
