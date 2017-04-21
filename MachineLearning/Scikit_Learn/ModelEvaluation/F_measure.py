import numpy as np
import sklearn.metrics as sm

'''
F-beta综合考虑了查准率和查全率，并且引入了额外的参数beta来控制两者谁占的比重更大，
具体原理参考.\Translation\ModelEvaluation
'''

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print('y_true: ', y_true)
print('y_pred: ', y_pred)
# 对macro来说，先计算0,1,2各自的f1_score，再取平均值
# P0 = 2/3, R0 = 1, F0 = 4/5 = 0.8
# P1 = 0... F1 = 0; P2 = 0... F2 = 0
# so Fmacro = (F0 + F1 + F2) / 3 = 0.8/3
fmacro = sm.f1_score(y_true, y_pred, average='macro')
print('macro f1_score: ', fmacro)

# 对micro来说，先计算0, 1, 2各自的Pmicro, Rmicro, 然后带入公式计算Fmicro
fmicro = sm.f1_score(y_true, y_pred, average='micro')
Pmicro = sm.precision_score(y_true, y_pred, average='micro')
Rmicro = sm.recall_score(y_true, y_pred, average='micro')
fmicro_me = 2 * Pmicro * Rmicro / (Pmicro + Rmicro)
print('micro f1 score: ', fmicro)
print('micro f1_score by myself: ', fmicro_me)
print('fmicro == fmicro_me: ', np.allclose(fmicro, fmicro_me))

# average = None 时，打印出每个label对应的f1_score
fnone = sm.f1_score(y_true, y_pred, average=None)
print('f1_score with average = None: ', fnone)

# beta = 0.5 时
fbeta_macro = sm.fbeta_score(y_true, y_pred, average='macro', beta=0.5)
fbeta_micro = sm.fbeta_score(y_true, y_pred, average='micro', beta=0.5)
fbeta_none = sm.fbeta_score(y_true, y_pred, average=None, beta=0.5)
print('beta: ', 0.5)
print('fbeta_macro score: ', fbeta_macro)
print('fbeta_micro score: ', fbeta_micro)
print('fbeta score with average = None: ', fbeta_none)
