import numpy as np
import sklearn.metrics as sm

'''
有些metric就是为二分类任务设计的，比如f1_score, roc_auc_score等，这类情况下，正例的标签默认都是1，但这个可以
通过pos_label参数来配置
'''

# 最简单的accuracy score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score = sm.accuracy_score(y_true, y_pred)
accuracy_score_unnormed = sm.accuracy_score(y_true, y_pred, normalize=False)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)
print('accuracy score: ', accuracy_score) # rate
print('unnormalized accuracy score: ', accuracy_score_unnormed) # No.
