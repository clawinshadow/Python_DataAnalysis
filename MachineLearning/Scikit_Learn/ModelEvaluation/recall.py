import numpy as np
import sklearn.metrics as sm

'''
在二分类问题中：
            tp
recall = ---------
          tp + fn
即所有正例中，被模型成功预测出来的正例之占比

在多分类问题中，与precision类似，计算recall需要指定平均的策略是micro还是macro
micro的话就是分别计算每个分类的tp, fn，到最后统一加总起来计算recall
macro的话就是分别计算出每个分类的recall，到最后取算术平均

pos_label参数可以指定正例是1还是0..or else
'''

# 二分类问题
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
recall = sm.recall_score(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)

# tp = 1, fn=1, 总共2个正例，有一个被预测出来了，所以recall为50%
print('recall: ', recall)
# 指定pos_label = 0, 此时recall应为1
recall = sm.recall_score(y_true, y_pred, pos_label=0)
print('recall when pos_label = 0: ', recall)

print('{0:-^70}'.format('Multilables'))

# 多分类问题
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
recall = sm.recall_score(y_true, y_pred, average='macro')
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)
# R0 = 1, R1 = 0, R2 = 0
# Pmacro = (R0 + R1 + R2)/3 = 1/3
print('macro recall: ', recall)
# label 0: tp = 2, fn = 1
# label 1: tp = 0, fn = 2
# label 2: tp = 0, fn = 1
# Pmicro = mean(tp) / [mean(tp) + mean(fp)] = (2/3) / (2/3 + 4/3) = 1/3
precision = sm.recall_score(y_true, y_pred, average='micro')
print('micro recall: ', recall)
