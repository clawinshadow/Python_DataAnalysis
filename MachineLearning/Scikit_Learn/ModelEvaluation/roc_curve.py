import numpy as np
import sklearn.metrics as sm

'''
roc 曲线的横轴和纵轴分别为FPR和TPR, TPR就是recall,
TPR: TP / (TP + FN)
FPR: FP / (FP + TN)

原理与precision_recall_curve差不多, 只不过thresholds不会剔除最小的一个
'''

y_true = np.array([1, 1, 2, 2])
probs_pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = sm.roc_curve(y_true, probs_pred, pos_label=2)
# 当threshold = 0.8时， pred = [1, 1, 1, 2]
# TP = 1, FN = 1, FP = 0, TN = 2
# TPR = 1/2, FPR = 0
# threshold = 0.4时，pred = [1, 2, 1, 2]
# TP = 1, FN = 1, FP = 1, TN = 1
# TPR = FPR = 0.5
print('y_true: ', y_true)
print('y_pred: ', probs_pred)
print('fpr: ', fpr)
print('tpr: ', tpr)
print('thresholds: ', thresholds)
