import numpy as np
import sklearn.metrics as sm

'''
混淆矩阵，在二分类问题中就是tp, fp, tn, fn, 在多分类问题中可以参考下面的例子：

                     Predicted
              Cat	Dog	Rabbit

	Cat	5	3	0
Actual
        Dog	2	3	1
class
        Rabbit	0	2	11

分别计算每个label在预测中有多少个，其中又各自有多少个预测对的和预测错的，形成一个n*n的矩阵(n个label)
'''

# 多分类问题
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cm = sm.confusion_matrix(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)

# label 0 在pred中总共有3个，有2个是对应true中的0, 0个对应true的1，1个对应true中的2
# label 1 在pred中总共有0个，有0个是对应true中的0, 0个对应true的1，0个对应true中的2
# label 2 在pred中总共有3个，有0个是对应true中的0, 1个对应true的1，2个对应true中的2
# 所以label 0的这一列为(2, 0, 1).T ........
print('confusion matrix: \n', cm)
print('{0:-^70}'.format('Binary Classfication'))

# 二分类问题，但是这个模块里面的matrix是按0，1来进行排序的，但一般1才是正例，所以tp,fp之类的顺序与常识不一致
y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
cm = sm.confusion_matrix(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)
print('confusion matrix: \n', cm)
tn, fp, fn, tp = sm.confusion_matrix(y_true, y_pred).ravel()
print('tn, fp, fn, tp: ',(tn, fp, fn, tp))
