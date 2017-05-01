import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

'''
决策树是一个不带参数的有监督学习方法，用于分类和回归。目标是根据数据集的属性来创建一个学习简单决策规则的模型，
用来预测未知的变量。关于决策树的优点和缺点列举如下：

优点：
    1. 很容易理解和解释，决策树能够可视化
    2. 准备很少的数据就够了。其它的很多技术经常要求数据的各种规范化，创建dummy的变量或者移除非法数据，简而言之
       就是数据预处理特别麻烦
    3. 使用决策树的时间代价(预测数据)是数据总量的对数，这个是非常快的
    4. 能够处理数值型和分类型的数据，其它算法往往只能处理特定的一种
    5. 能处理多元输出的问题 multi-output
    6. 使用了一个白盒模型(white box)，在白盒模型中，只要我们成功建立了这个模型，我们就能看清里面的具体逻辑，
       而在黑盒模型比如神经网络中，我们只能得到一个预测结果，而不知道为什么会是这样。
    7. 可以用统计量的测试来验证这个模型，这样就使得计算魔性的可靠性变为可能
    8. 即便假设和实际模型背离，它依然表现良好

缺点：
    1. 决策树往往容易训练出过分复杂的模型，这样对数据的泛化就不会很好，这被称之为过拟合overfitting. 需要使用
       剪枝(当前sklearn还不支持)，设定叶节点的最小样本数或者设定树的最大深度等方法来避免这个问题
    2. 决策树不太稳定，有时候数据的微小变化就会产生一个完全不同的树。在集成学习中使用决策树能减缓这个问题
    3. 要找到一个最优化的决策树实际上一个NP-困难的问题，我们在实际中用到的决策树学习算法通常是启发式算法，
       比如在每个节点上选择当前的最优解这种贪心算法。那么显然我们得到的最终解有时候并不是全局最优解，同样，
       可以使用集成学习来缓解这个问题
    4. 有些概念对决策树来说很难表达，比如异或，多路转换问题等等，XOR用神经网络很好解决
    5. 如果某些分类在数据集中占据主要地位的话，生成的决策树会是有偏的，所以在训练数据前最好平衡一下数据集。
'''

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print('X_train sample: \n', X_train[:5])
print('X_test sample: \n', X_test[:5])
print('y_train sample: \n', y_train[:2])
print('y_test sample: \n', y_test[:2])

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)

# what's the tree_ attribute of the DecisionTreeClassifier
# node_count表示树中所有的节点数，分为test node和leaf node，test node是if-then-else的分支节点
# children_left 存储每个节点的左子节点id, 叶节点的左子节点id为-1，总数量与node_count一致
# children_right 的逻辑与left一样，只不过存储的是右子节点
# feature表示每个test node的值需要取自数据集的哪个属性，第几列
# threshold表示每个test node上的阈值，小于threshold则去往左子节点，大于则去往右子节点
# 除了Node_count是整数之外，其余的都是数组，并且数组长度 = node_count
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
print('tree_ attribute: ', estimator.tree_)
print('node_count: ', n_nodes)
print('children_left: ', children_left)
print('children_right: ', children_right)
print('feature: ', feature)
print('threshold: ', threshold)

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has {0} nodes and has the following tree structure:".format(n_nodes))
for i in range(n_nodes):
    if is_leaves[i]:
        print("{0}node={1} leaf node.".format(int(node_depth[i]) * "\t", i))
    else:
        print("{0}node={1} test node: go to node {2} if X[:, {3}] <= {4}s else to {5}.".format(\
            int(node_depth[i]) * "\t", i, children_left[i], feature[i], threshold[i], children_right[i]))

# decision_path, 形如以下的结果：
#  (0, 0)	1
#  (0, 2)	1
#  (0, 4)	1
#  (1, 0)	1
#  (1, 2)	1
#  (1, 3)	1
#  ......
# 表示测试集中的数据，第0个测试数据的路径为 0 -> 2 -> 4, 第1个的路径为 0 -> 2 -> 3
dp1 = estimator.decision_path(np.array([X_test[0]])) # 某一个测试集数据的决策路径
node_indicator = estimator.decision_path(X_test)     # 所有测试集的决策路径矩阵
print('X_test[0]: ', X_test[0])
print('decision_path of X_test[0]: \n', dp1)
print('X_test: ', X_test)
print('decision ath of all X_test : \n', node_indicator)

# apply方法得出所有的分类最终结果，走到哪个最终的叶子结点
leave_id = estimator.apply(X_test)
print('destination of all decision_paths: \n', leave_id)

# predict方法得出的是真实意义上的分类结果，而apply方法只是得出的所有叶子结点的索引
# 但它们两个是可以一一对应上的
predictResult = estimator.predict(X_test)
print('Predict classification result of X_test: \n', predictResult)

predictProbas = estimator.predict_proba(X_test)
print('Predict probabilities of X_test: \n', predictProbas)

# 这个score就是精确率，将预测值与真实值逐一对比，吻合的/总数 即可
print('real labels: \n', y_test)
print('score: ', estimator.score(X_test, y_test))
