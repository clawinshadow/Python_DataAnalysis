import numpy as np
import matplotlib.pyplot as plt

def L2(r):
    return r**2

def L1(r):
    return np.abs(r)

def huberLoss(r, delta):
    '''
    delta相当于一个门槛，小于delta的，损失函数是L2的，大于delta的，损失函数是L1的。
    HuberLoss的优点在于它组合了两种惩罚力度L2和L1，并且它在全定义域内都是可导的，所以可以很方便的进行最优化求解
    
                      |- r**2/2,         if |r| <= δ
    HuberLoss(r, δ) = |
                      |- δ|r|- δ**2/2,   if |r| >  δ

    '''
    result = []
    for i in range(len(r)):
        if np.abs(r[i]) <= delta:
            result.append(0.5 * r[i]**2)                             # 关于r的二次函数，L2                        
        else:
            result.append(delta * np.abs(r[i]) - 0.5 * delta**2)  # 关于r的线性函数，L1
            
    return np.array(result)

delta = 2    # 根据书里面的图形观测出来的
r = np.linspace(-3, 3, 300)
loss_L1 = L1(r)
loss_L2 = L2(r)
loss_huber = huberLoss(r, delta)

fig = plt.figure(figsize=(7, 6))
fig.canvas.set_window_title('huberLossDemo')
plt.axis([-3, 3, -0.5, 5])
plt.xticks(np.arange(-3, 3.2, 1))
plt.yticks(np.arange(-0.5, 5.1, 0.5))
plt.plot(r, loss_L2, 'r-', lw=2, label='L2')
plt.plot(r, loss_L1, color='midnightblue', lw=2, ls=':', label='L1')
plt.plot(r, loss_huber, 'g-.', lw=2, label='huber')

plt.legend(loc='upper center')
plt.show()
