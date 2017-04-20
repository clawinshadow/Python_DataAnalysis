import numpy as np
import sklearn.metrics as sm

'''
这个统计量最早不是为分类问题设计的，它用来评估两个评委对同一份样本评选结果的一致性(inter-annotator agreement)
，设想有50份论文交由两位评委A和B去评审，结果可以为'Yes'或者'No'，如果A和B各自的评审结果如下表所示：

              B
          Yes   No
A   Yes   20    5
     No   10    15

则我们可以计算精确度 Po: (20 + 15) / 50 = 0.7
然后计算另一个量 Pe:
    Pyes = A.Pyes * B.Pyes = [(20 + 5) / 50] * [(20 + 10) / 50] = 0.5 * 0.6 = 0.3
    Pno = A.Pno * B.Pno = [(10 + 15) / 50] * [(5 + 15) / 50] = 0.5 * 0.4 = 0.2
    Pe = Pyes + Pno = 0.5
最后我们计算这个kappa统计量：
    kappa = (Po - Pe) / (1 - Pe) = 0.4

衍生到多分类问题中，我们也是先计算总的精确度Po，然后再计算各个label的P值Pi，然后加总所有的Pi值得到Pe，最后计算
出Kappa。关键在于计算各个label的Pi值，需要分别计算当前label在y_true和y_pred中所占的比重再相乘即可
'''

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
kappa = sm.cohen_kappa_score(y_true, y_pred)
print('sample labels: ', y_true)
print('predicted labels: ', y_pred)

# Po = 4/6 (4个预测对了)
# P0 = 2/6 * 3/6 = 1/6 (0在y_true中占比2/6, 在y_pred中占比3/6)
# P1 = 1/6 * 0 = 0
# P2 = 3/6 * 3/6 = 1/4
# Pe = P0 + P1 + P2 = 5/12
# kappa = (Po - Pe) / (1 - Pe) = 3/7
print('cohen\'s kappa score: ', kappa)
print('kappa == 3/7: ', np.allclose(kappa, 3/7))
