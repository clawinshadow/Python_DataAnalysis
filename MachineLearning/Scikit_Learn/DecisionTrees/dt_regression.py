import numpy as np
import sklearn.tree as st
import sklearn.preprocessing as sp
import pydotplus

'''
关于决策树的回归，详细的tutorial见这个URL：http://www.saedsayad.com/decision_tree_reg.htm

sklearn中的决策树不能做string类型的处理，每一个test node都是数值型的二分类问题，这个是个
巨大的缺陷，待考证
'''

feature_names = ['Outlook', 'Temp', 'Humidity', 'Windy']
target_name = ['Hours played']
train_raw = [['Rainy', 'Hot', 'High', 'False'],
             ['Rainy', 'Hot', 'High', 'True'],
             ['Overcast', 'Hot', 'High', 'False'],
             ['Sunny', 'Mild', 'High', 'False'],
             ['Sunny', 'Cool', 'Normal', 'False'],
             ['Sunny', 'Cool', 'Normal', 'True'],
             ['Overcast', 'Cool', 'Normal', 'True'],
             ['Rainy', 'Mild', 'High', 'False'],
             ['Rainy', 'Cool', 'Normal', 'False'],
             ['Sunny', 'Mild', 'Normal', 'False'],
             ['Rainy', 'Mild', 'Normal', 'True'],
             ['Overcast', 'Mild', 'High', 'True'],
             ['Overcast', 'Hot', 'Normal', 'False'],
             ['Sunny', 'Mild', 'High', 'True']]
target = [26, 30, 48, 46, 62, 23, 43, 36, 38, 48, 48, 62, 44, 30]

train = np.array(train_raw)
le = sp.LabelEncoder()
train_encoded = np.vstack((le.fit_transform(train[:, 0]),
                           le.fit_transform(train[:, 1]),
                           le.fit_transform(train[:, 2]),
                           le.fit_transform(train[:, 3]))).T
print(train_encoded)

clf = st.DecisionTreeRegressor(criterion='mse', max_depth=3)
clf = clf.fit(train_encoded, np.transpose(target))
dot_data = st.export_graphviz(clf, out_file=None, 
                         feature_names=feature_names, 
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_png("regression.png")
