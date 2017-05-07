import numpy as np
import sklearn.tree as st
import sklearn.preprocessing as sp
import pydotplus
import matplotlib.pyplot as plt

'''
关于决策树的回归，详细的tutorial见这个URL：http://www.saedsayad.com/decision_tree_reg.htm

sklearn中的决策树不能做string类型的处理，每一个test node都是数值型的二分类问题，这个是个
巨大的缺陷，待考证
'''

def sample1():
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
    graph.write_png("dt_regression_sample1.png")

def plotfigure(X, X_test, y, yp):
    plt.figure()
    plt.scatter(X, y, c='k', label='traing data')                 # X, y 是训练数据集的散点图
    plt.plot(X_test, yp, c='r', label='max_depth=5', linewidth=2) # X_test是测试集，对应的yp是回归值
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.show()

def sample2():
    x = np.linspace(-5, 5, 200)
    siny = np.sin(x)                                # 生成训练集
    X = np.transpose(x)[:, np.newaxis]
    print('X.shape: ', X.shape)
    y = siny + np.random.rand(1, len(siny)) * 1.5   # 假如噪声点集
    print(y.shape)
    y = y[0, :]
    print(y.shape)
    # Fit regression tree
    clf = st.DecisionTreeRegressor(max_depth=4)
    fitmodel = clf.fit(X, y)                        # 必须保证X为2维数组(m, n), y为一维数组，并且len(y) = n

    # Visualize the decision tree
    dot_data = st.export_graphviz(fitmodel, out_file=None, 
                             feature_names=['data'], 
                             filled=True, rounded=True,  
                             special_characters=True) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_png("dt_regression_sample2.png")

    # Predict
    X_test = np.arange(-5, 5, 0.05)[:, np.newaxis]
    yp = clf.predict(X_test)

    plotfigure(X, X_test, y, yp)

sample2()
    
