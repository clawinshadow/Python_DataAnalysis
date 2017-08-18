import numpy as np
import matplotlib.pyplot as plt

'''
demos about softmax function, 带一个温度参数T，T越小，softmax越倾向于极端的分布，即hard-max. T越大，max的作用越弱，倾向于
均匀分布 S(η/T)
'''

def softmax(eta, T):
    eta = eta / T
    total = np.sum(np.exp(eta))
    result = np.divide(np.exp(eta), total)

    return result

fig = plt.figure(figsize=(9, 8))

def Draw(fig, subplotIndex, T):
    x = [1, 2, 3]
    eta = np.array([3, 0, 1])
    probs = softmax(eta, T)

    ax = fig.add_subplot(subplotIndex)
    ax.bar(x, probs, align='center')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_title('T = {0}'.format(T))

Draw(fig, 221, 100)
Draw(fig, 222, 1)
Draw(fig, 223, 0.1)
Draw(fig, 224, 0.01)

plt.show()
