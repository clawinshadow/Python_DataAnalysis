import operator
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def Draw(index, N):
    theta = 0.7               # 所有的样本都来自 Ber(θ = 0.7)
    ber = ss.bernoulli(theta)

    iterCounts = 10000        # 抽样10000次，每次10个样本或者100个样本
    N = N

    data = {}
    for i in range(iterCounts):
        sample = ber.rvs(N)
        # 当N很大时，为了避免生成过多的bar，一律只保留小数点后一位
        MLE = np.around(np.count_nonzero(sample) / len(sample), 1)  
        if MLE not in data:
            data[MLE] = 1
        else:
            data[MLE] += 1

    print('sample distribution when N = {0}: {1}'.format(N, data))
    MLE = sorted(iter(data.items()), key=operator.itemgetter(1), reverse=True)[0][0]

    print(list(data.keys()), list(data.values()))
    plt.subplot(index)
    plt.bar(list(data.keys()), list(data.values()), width=0.1, align='center', color='midnightblue', edgecolor='k')
    plt.xlim(0, 1.1)
    plt.title('Boot: true = {0}, N = {1}, MLE = {2}, se = 0.001'.format(theta, N, MLE))

plt.figure(figsize=(11, 5))
Draw(121, 10)
Draw(122, 100)

plt.show()
