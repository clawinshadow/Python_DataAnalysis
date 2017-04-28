import numpy as np
import scipy.stats as ss

'''
demos about Expectation Maximisation(EM) algorithm.
EM 算法是一个比较复杂的数据分析算法，查阅了很多资料，还是这个例子比较浅显易懂
设想我们有一组样本数据点，里面包含两组数据，分别从不同的总体里面抽取出来的，这意味着它们各自的概率分布也是不同的。
假设分别用红色和蓝色来标识出两组不同的数据点，并且假设两个总体都服从正态分布，只不过各自的参数不一样：

已知样本数据 X: (R, B, B, R, B, R, R, R, B, B, R, R...)
未知总体参数：(μ1, σ1), (μ2, σ2)
这种情况下如果要估计样本参数，我们可以很简单的使用MLE来估计，分别计算R和B的均值和标准差即可

但是如果我们抹去样本数据点的颜色，变为：(X X X X .... X)呢，这时候还能估计出样本参数吗？
答案是可以的，期望最大化算法(EM)可以解决这个问题，核心思想是迭代，如下：
1. 先给个参数的初始值，随机的估计 -> Θ1:(μ1, σ1, μ2, σ2)
2. 给定了模型参数θ1，就可以计算出每个数据点属于R或者是B的可能性(likelihood)了，这一步就是Expectation
3. 基于每个数据点的可能性计算两个分组R和B对于当前数据点的权重，即将每个数据点分解为两个数据对(Xr, Xb)
4. 将这些数据整合在一起，整合所有的(Xr, Xb)重新计算出模型参数的最好估计Θ2，这一步就是Maximisation
5. 将Θ2作为输入参数代入第二步，重复执行2-4步，直到参数收敛至满意的结果

已经有很多论文证明过，EM算法确定会收敛到局部最优解
'''

def CalcWeights(likelihood_of_color, likelihood_total):
    '''
    将两个比值标准化到区间(0, 1)里面，比如 0.1/0.2 = 0.5 / 1, 0.5就是权重
    '''
    return likelihood_of_color / likelihood_total

def estimate_mean(data, weight):
    return np.sum(data * weight) / np.sum(weight)

def estimate_std(data, weight, mean):
    var = np.sum(weight * (data - mean)**2) / np.sum(weight)
    return np.sqrt(var)

def EM_Iterate(parameters, sample):
    # step 1: 获取参数初始值，或是上一次迭代所得的结果作为这一次迭代的输入
    red_mean_old = parameters[0]
    red_std_old = parameters[1]
    blue_mean_old = parameters[2]
    blue_std_old = parameters[3]

    # step 2: 计算每个数据点属于红色或者蓝色的可能性，用概率密度pdf来计算，cdf是不行的
    likelihood_of_red = ss.norm(red_mean_old, red_std_old).pdf(sample)
    likelihood_of_blue = ss.norm(blue_mean_old, blue_std_old).pdf(sample)

    # step 3: 将每个数据点对应的两个likelihood转化为权重，再按权重来分解每个样本数据点
    #         权重的本质实际上是每个数据点属于R还是B的倾向性，
    likelihood_total = likelihood_of_red + likelihood_of_blue
    weights_of_red = CalcWeights(likelihood_of_red, likelihood_total)
    weights_of_blue = CalcWeights(likelihood_of_blue, likelihood_total)

    # step 4: 根据红色和蓝色的权重数组，重新计算模型参数
    red_mean_new = estimate_mean(sample, weights_of_red)
    red_std_new = estimate_std(sample, weights_of_red, red_mean_old)
    blue_mean_new = estimate_mean(sample, weights_of_blue)
    blue_std_new = estimate_std(sample, weights_of_blue, blue_mean_old)
    print('Population Params by EM(red_mean, red_std, blue_mean, blue_std): \n',\
      red_mean_new, red_std_new, blue_mean_new, blue_std_new)

    return red_mean_new, red_std_new, blue_mean_new, blue_std_new
    

# 设置模型参数, Red ~ N(3, 0.8), Blue ~ N(7, 2)
red_mean = 3
red_std = 0.8
blue_mean = 7
blue_std = 2

# 生成样本数据
np.random.seed(110) # make it reproducible
size = 20
red = np.random.normal(red_mean, red_std, size=size)      # 生成红色数据点
blue = np.random.normal(blue_mean, blue_std, size=size)   # 生成蓝色数据点
sample = np.sort(np.concatenate((red, blue)))             # 将两组数据点糅合在一起，形成样本
print('red points: ', red)
print('blue points: ', blue)
print('Sample: ', sample)

# 用极大似然法估MLE来估计样本参数，此时假定我们知道各数据点的颜色
red_mean_guess = np.mean(red)
red_std_guess = np.std(red)
blue_mean_guess = np.mean(blue)
blue_std_guess = np.std(blue)
print('Population Params by MLE(red_mean, red_std, blue_mean, blue_std): ',\
      red_mean_guess, red_std_guess, blue_mean_guess, blue_std_guess)
print('Real Population Params: ', red_mean,  red_std, blue_mean, blue_std)

# 抹去数据点的颜色，用EM算法来估计总体参数，迭代5次
iteCount = 20
params_initial = (1.1, 2, 9, 1.7)
for i in range(iteCount):
    print('{0:-^70}'.format('Iteration ' + str(i)))
    params_initial = EM_Iterate(params_initial, sample)
