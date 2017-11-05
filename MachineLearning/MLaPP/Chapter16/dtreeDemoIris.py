import pydotplus
import numpy as np
import sklearn.datasets as sd
import sklearn.tree as st
import sklearn.model_selection as sms
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
算法的思想比较简单，实现起来大概是用递归来分叉，以前写过，本例中就不再写了，就使用sklearn里面的方法，很好使
tips: 在CV中，需要返回training score的时候就用cross_validate方法
'''

# load dataset Iris
iris = sd.load_iris()
print(iris.keys())
X = iris['data']
y = iris['target']
feature_names = iris['feature_names']
class_names = iris['target_names']
print(X.shape, y.shape)
print(feature_names, '\n', class_names)
X = X[:, :2]     # only use the first 2 features, sepal length & sepal width
feature_names = feature_names[:2]
print(np.unique(y))

# plot dataset
fig = plt.figure(figsize=(10, 8))
fig.canvas.set_window_title('dtreeDemoIris_1')

ax1 = plt.subplot(221)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.axis([4, 8, 2, 4.5])
plt.xticks(np.linspace(4, 8, 9))
plt.yticks(np.linspace(2, 4.5, 6))
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.plot(X[y == 0][:, 0], X[y == 0][:, 1], 'ro', fillstyle='none', linestyle='none', label='setosa')
plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], 'gs', fillstyle='none', linestyle='none', label='versicolor')
plt.plot(X[y == 2][:, 0], X[y == 2][:, 1], 'bd', fillstyle='none', linestyle='none', label='virginica')
plt.legend()

# Fit with Decistion Tree
# Attention: 1. 本例中iris_unpruned.png展示的图形与书中的是完全一样的，
#               但是预测的图形有几个点不一样，是因为有部分父节点，分叉的 < 和 <= 的区别
#            2. min_samples_split 用于决定分叉的逻辑，如果一个节点的数目小于这个参数的值，
#               则即便不纯也不再分叉了
#            3. max_depth 显而易见是用来控制树的深度
#            4. max_leaf_nodes 是对应于书中的节点来设置的，书中是19个，这里max就设置成20，用来merge多余的叶节点
clf = st.DecisionTreeClassifier(min_samples_split=10, max_depth=10, max_leaf_nodes=20).fit(X, y)
dot_data = st.export_graphviz(clf, out_file=None, feature_names=feature_names, \
                              class_names=class_names, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris_unpruned.png")

xx, yy = np.meshgrid(np.linspace(4, 8, 41), np.linspace(2, 4.5, 26))
xtest = np.c_[xx.ravel(), yy.ravel()]
zz = clf.predict(xtest)

# Fit with CV, for prune the tree
def one_zero_loss(ground_truth, predictions):
    return np.mean(ground_truth != predictions)

loss = sm.make_scorer(one_zero_loss, greater_is_better=True)
node_counts = np.linspace(2, 19, 18)  # this is max_leaf_nodes
K = 10
kf = sms.KFold(n_splits=K, shuffle=False)
trainingScores = np.zeros((len(node_counts), K))
testScores = np.zeros((len(node_counts), K))
for i in range(len(node_counts)):
    max_terminal_nodes = (int)(node_counts[i])
    dtc = st.DecisionTreeClassifier(min_samples_split=10, max_depth=10, max_leaf_nodes=max_terminal_nodes)
    cvResult = sms.cross_validate(dtc, X, y, cv=kf, scoring=loss, return_train_score=True)
    trainingScores[i] = cvResult['train_score']
    testScores[i] = cvResult['test_score']

scores_train = np.mean(trainingScores, axis=1)
scores_test = np.mean(testScores, axis=1)
print(scores_train)
print(scores_test)
optimalIndex = np.argmin(scores_test)
print('optimal max terminal nodes: ', node_counts[optimalIndex])

stdErr = np.std(scores_test, ddof=1)  # biased standard variance
cv_se = scores_test[optimalIndex] + stdErr

optimal_dtc = st.DecisionTreeClassifier(min_samples_split=10, max_depth=10, \
                                        max_leaf_nodes=(int)(node_counts[optimalIndex])).fit(X, y)
zz_pruned = optimal_dtc.predict(xtest)
dot_data2 = st.export_graphviz(optimal_dtc, out_file=None, feature_names=feature_names, \
                              class_names=class_names, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data2)
graph.write_png("iris_pruned.png")

def plotPrediction(index, title, data):
    ax2 = plt.subplot(index)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.axis([3.8, 8.2, 1.9, 4.6])
    plt.xticks(np.linspace(4, 8, 9))
    plt.yticks(np.linspace(2, 4.5, 6))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.plot(xtest[data == 0][:, 0], xtest[data == 0][:, 1], 'ro', fillstyle='none', linestyle='none', mew=0.5,
             label='setosa')
    plt.plot(xtest[data == 1][:, 0], xtest[data == 1][:, 1], 'gs', fillstyle='none', linestyle='none', mew=0.5,
             label='versicolor')
    plt.plot(xtest[data == 2][:, 0], xtest[data == 2][:, 1], 'bd', fillstyle='none', linestyle='none', mew=0.5,
             label='virginica')
    plt.legend()

# plot the unpruned prediction
plotPrediction(222, 'unpruned decistion tree', zz)

# plot the CV results
plt.subplot(223)
plt.axis([0, 20, 0.1, 0.8])
plt.xlabel('Number of terminal nodes')
plt.ylabel('Cost (misclassification error)')
plt.xticks(np.linspace(0, 20, 5))
plt.yticks(np.linspace(0.1, 0.8, 8))
plt.plot(node_counts, scores_test, color='midnightblue', lw=2, label='Cross-validation')
plt.plot(node_counts, scores_train, 'r--', lw=2, label='Training set')
plt.plot(node_counts[optimalIndex], scores_test[optimalIndex], color='purple', marker='o',\
         markersize=10, fillstyle='none', linestyle='none', label='Best choice')
plt.axhline(cv_se, color='k', linestyle=':', lw=2, label='Min + 1 std. err.')
plt.legend()

# plot the pruned prediction
plotPrediction(224, 'pruned decistion tree', zz_pruned)

plt.tight_layout()

# plot the unpruned tree
# 这里有个缺点就是它不会merge tree的子节点，导致有多个冗余的分支，图像里面可以轻易看出来
# 虽然预测的结果是一样的，但是图像就看起来很复杂，没有书里面的简洁
fig2 = plt.figure(figsize=(11, 7))
fig2.canvas.set_window_title('dtreeDemoIris_UnprunedTree')
plt.subplot()
image = mpimg.imread("iris_unpruned.png")
plt.axis('off')
plt.imshow(image)
plt.tight_layout()

# plot the pruned tree
fig3 = plt.figure(figsize=(6, 5))
fig3.canvas.set_window_title('dtreeDemoIris_prunedTree')
plt.subplot()
image = mpimg.imread("iris_pruned.png")
plt.axis('off')
plt.imshow(image)
plt.tight_layout()

plt.show()