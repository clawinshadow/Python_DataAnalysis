import numpy as np
import sklearn.metrics as sm

'''
在二分类以及多分类问题中，杰卡德相似系数等价于精确度
'''

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
js = sm.jaccard_similarity_score(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)
print('jaccard similarity score: \n', js)
