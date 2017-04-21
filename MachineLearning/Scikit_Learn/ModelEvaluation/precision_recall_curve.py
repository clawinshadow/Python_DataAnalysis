import numpy as np
import sklearn.metrics as sm

'''
这个不太好理解，一般我们计算precision和recall是有个明确的预测分类，但是有些情况下比如神经网络，
它预测出来的结果是一个概率值比如0.7，这样一来我们必须设定某个阈值threshold，比如0.5，大于该阈值
的视为分类1，小于它的视为0.

precision_recall_curve就是给定一堆阈值，以及预测出来的概率值，对每个阈值计算其precision和recall，
然后可以使用这些数据将其可视化出来，就是P-R曲线

仅可用于二分类问题
'''

# 4个样本的真实分类
y_true = np.array([0, 0, 1, 1])
# 4个样本预测出来的概率值
probs_pred = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = sm.precision_recall_curve(y_true, probs_pred)
print('y_true: ', y_true)
print('probs_pred: ', probs_pred)
# thresholds是该方法自动生成的，一般式np.unique(probs_pred),但要剔除其中最小的一个
print('thresholds: ', thresholds)
# 对应每个threshold，计算precisioin和recall
# 当threshold=0.4时，y_pred = [0, 1, 1, 1] => precision = 2/3, recall = 1
# 即大于等于threshold的，都是正例1
# 但要注意最后一个precision值必定是1，并且么有对应的阈值
# 同理recall最后一个值为0， 也没有对应的阈值
# 是为了在图形展示的时候形成一个点(0, 1)，让图形从Y轴上的这个点出发，更好看一点(personally)
print('precision: ', precision)
print('recall: ', recall)

print('{0:-^70}'.format('Seperate Line'))

# 与之相对应的还有一个较重要的方法就是average_precision_score
# 用来根据不同的平均策略，来计算micro或者macro的平均精度
aps = sm.average_precision_score(y_true, probs_pred, average='macro')
print('y_true: ', y_true)
print('probs_pred: ', probs_pred)
print('average precision score with macro stratege: ', aps)
print('macro aps == np.mean(precision): ', np.allclose(aps, np.mean(precision)))
# 对于micro，需要计算每个threshold对应的tp, fp
aps = sm.average_precision_score(y_true, probs_pred, average='micro')
print('average precision score with micro stratege: ', aps)

