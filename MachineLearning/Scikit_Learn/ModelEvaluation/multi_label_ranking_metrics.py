import numpy as np
import sklearn.metrics as sm

'''
这里介绍一个概念Multi-label classification, 姑且称之为多标签分类吧，这个和我们以前所熟知的多分类问题是不一样的,

多分类问题是Multiclass classification, 比如将一组西瓜分为[优等瓜，好瓜，中等瓜，坏瓜]，对每个样本来说，
输出的分类标签不再是[0, 1](简单的二分类问题)，而是四种瓜中的一个，但它依然是一个标量。

而多标签分类是什么概念呢，对每个样本来说，它的输出不再是简单的一个标量，而是一个二分类的向量，比如[0, 1, 1, 0, 0]

那么对于多分类问题来说，多个样本的输出是一个一维数组，比如: [1, 2, 4, 2, 3, 1, 1, 2...]
而对于多标签分类来说，多个样本的输出会是一个二维数组，比如：[[0, 1, 1, 1], [1, 0, 1, 0]...]
习惯上我们将多标签分类的输出的shape定义为 (Nsamples * Nlabels), 行数为sample的数量，列数为labels的数量

1. Coverage Error:
   这是最简单的一个度量，假设输入某个关键字，输出某些labels，按概率从大到小排序，得到的结果如下：
   Query	Results	Correct       response	 Rank	 Reciprocal rank
     cat	catten, cati, cats	cats	  3	   1/3
    tori	torii, tori, toruses	tori	  2	   1/2
    virus	viruses, virii, viri	viruses	  1	    1

   Coverage Error就是一种对模型预测能力的度量，分别计算每个样本预测的labels中，正确结果排在第几位然后取平均。
   比如第一个cat样本，正确的结果应该是cats，但仅排在第3位，第二个排在第2， 第三个排第1, so：

   Coverage Error = (3 + 2 + 1) / 3 = 2

   数学定义比较复杂...假设labels属于{0, 1}(Nsamples*Nlabels), 每个labels矩阵关联一个对应的概率矩阵，shape一样
   那么计算每一行中每个输出为1的概率排名取最大值，即Max(Rank[ij]{Yij = 1}), 最后取所有每行的值的平均值
   
2. Label ranking average precision(LRAP)
   这个度量即是上一个CoverageError的调和平均，含义差不多的，只不过算法不一样
   这个LRAP的值域是[0, 1], 越大表示模型的预测能力越好
   
3. Ranking loss
   这个是最复杂的一个度量，值域在[0, 1]之间，因为是loss，所以当然是越小越好，0表示完美的预测
   难在计算每一行的loss值：

   loss per sample = |{k, l}: f[i, k] < f[i, l], y[i, k] = 1, y[i, l] = 0| / [|yi|*(Nlabels - |yi|)

   对每一行来说，即每个样本来说，|yi|即该样本输出的labels数组中总共有几种不一样的分类，一般都是2，因为只有0或者1
   Nlabels对整个计算过程来说都是一样的，即labels的数量

   前面的半边表示要在这一行中选出一个(0, 1)对，并且要求1的概率小于0的概率，看能找出几个来就是几
   最后将每一行的loss per sample 加起来算个平均值即可

查阅了很多维基百科才得出上面的一点经验:
https://en.wikipedia.org/wiki/Multi-label_classification
https://en.wikipedia.org/wiki/Mean_reciprocal_rank   
'''

y_lables = np.array([[1, 0, 0], [0, 0, 1]])          # 每一个样本的输出lables
y_probas = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]]) # 每一个输出的概率，根据概率进行排序
ce = sm.coverage_error(y_lables, y_probas)
# 对第一个样本来说，即第一行，只有一个正例1，它对应的概率为0.75，排在第二位，那么该行的值为2
# 对第二个样本来说，也只有一个正例1，它的概率最小，为0.1， 排在第三位，该行的值为3
# 最后取平均，因为Nsamples = 2, 所以coverage error = (3 + 2) / 2 = 2.5
print('labels: \n', y_lables)
print('probabilities of each label: \n', y_probas)
print('(Nsamples, Nlables):', np.shape(y_lables))
print('Coverage Error: ', ce)
# 若将y_labels修改为如下所示，则第一行就有两个1了，一个排第2，一个排第3，我们取最大值则需要取3
y_lables = np.array([[1, 1, 0], [0, 0, 1]])  
ce = sm.coverage_error(y_lables, y_probas)
print('labels: \n', y_lables)
print('Coverage Error with : ', ce)  # (3 + 3) / 2

print('{0:-^70}'.format('LRAP'))

# 计算LRAP, 就是将Coverage Error中的各中间值按调和平均进行计算
y_lables = np.array([[1, 0, 0], [0, 0, 1]])    
# LRAP = 1/2 * (1/3 + 1/2) = 5/12 = 0.4166....
lrap = sm.label_ranking_average_precision_score(y_lables, y_probas)
print('labels: \n', y_lables)
print('probabilities of each label: \n', y_probas)
print('Label ranking average precision: ', lrap)

print('{0:-^70}'.format('Ranking Loss'))

# 对第一个样本来说，k只能取0，因为只有y[0, 0] = 1，
# 继而l也只能取2了，因为只有f[0, 0] < f[0, 2](0.75 < 1), 所以第一个样本的ranking loss为1
# 对第二个样本来说，k只能取2，只有y[1, 2] = 1, l可以取0和1都行，因为他们的概率都大于f[1, 2], 所以这一行为2
# 因为labels只包含0和1，所以|yi| = 2， labels总共有3个，所以Nlabels = 3
# 那么最终结果Ranking Loss = (1 / [2 * (3 - 2)] + 2 / [2 * (3 - 2)]) / 2 = 3/4 = 0.75
rl = sm.label_ranking_loss(y_lables, y_probas)
print('labels: \n', y_lables)
print('probabilities of each label: \n', y_probas)
print('Ranking loss: ', rl)

# perfect and minimal loss
y_probas = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
rl = sm.label_ranking_loss(y_lables, y_probas)
print('probabilities of each label: \n', y_probas)
print('Ranking loss: ', rl)


