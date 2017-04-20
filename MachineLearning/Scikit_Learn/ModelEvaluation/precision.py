import numpy as np
import sklearn.metrics as sm

'''
这个就不说了，最重要的几个评估指标之一
在二分类问题中：
               tp
precision = ---------
             tp + fp
即评估的所有正例中，真正例的占比

在多分类问题中，计算precision需要指定平均的策略是micro还是macro
micro的话就是分别计算每个分类的tp, fp，到最后统一加总起来计算precision
macro的话就是分别计算出每个分类的precision，到最后取算术平均

pos_label参数可以指定正例是1还是0..or else
'''

# 二分类问题
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
precision = sm.precision_score(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)

# tp = 1, 预测了一个为1，是真正例，所以precision为100%
print('precision: ', precision)
# 指定pos_label = 0, 此时precision应为2/3
precision = sm.precision_score(y_true, y_pred, pos_label=0)
print('precision when pos_label = 0: ', precision)

print('{0:-^70}'.format('Multilables'))

# 多分类问题
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
precision = sm.precision_score(y_true, y_pred, average='macro')
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)
# P0 = 2/3, P1 = 0, P2 = 0
# Pmacro = (P0 + P1 + P2)/3 = 2/9
print('macro precision: ', precision)
# label 0: tp = 2, fp = 1
# label 1: tp = 0, fp = 2
# label 2: tp = 0, fp = 1
# Pmicro = mean(tp) / [mean(tp) + mean(fp)] = (2/3) / (2/3 + 4/3) = 1/3
precision = sm.precision_score(y_true, y_pred, average='micro')
print('micro precision: ', precision)
