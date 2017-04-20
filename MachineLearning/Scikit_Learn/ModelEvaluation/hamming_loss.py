import numpy as np
import sklearn.metrics as sm

'''
hamming loss, 这个非常简单，将pred与true相比，相同的记为1，不同的记为0，然后将所有的值加起来，
再除以所有的样本总数即可。
'''

y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
hl = sm.hamming_loss(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)
# 只有第一个不一致，那么 hl = (1 + 0 + 0 + 0) / 4
print('hamming loss: ', hl)

print('{0:-^70}'.format('Seperate Line'))

y_pred = np.zeros((2, 2))
y_true = np.array([[0, 1], [1, 1]])
hl = sm.hamming_loss(y_true, y_pred)
print('sample labels: \n', y_true)
print('predicted labels: \n', y_pred)
print('hamming loss: ', hl)
