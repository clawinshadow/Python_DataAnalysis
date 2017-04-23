import numpy as np
import sklearn.metrics as sm

'''
Matthews correlation coefficient:
MCC = (tp*tn - fp*fn)/math.sqrt[(tp+fp)(tp+fn)(tn+fp)(tn+fn)]

MCC的值域在[-1, 1]之间，+1表示完美的预测，0表示随机的预测，-1表示完全逆反的预测
'''

y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
# tp = 2, fp = 1, fn = 1, tn = 0
# mcc = -1/math.sqrt(3*3*1*1) = -0.33333
mcc = sm.matthews_corrcoef(y_true, y_pred)
print('y_true: ', y_true)
print('y_pred: ', y_pred)
print('mcc: ', mcc)
