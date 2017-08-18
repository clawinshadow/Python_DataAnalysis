import numpy as np
import scipy.io as sio
import scipy.stats as ss
import matplotlib.pyplot as plt

# 注意这个方法，使用一些持久化了的样本数据集的时候会经常用到
mat = sio.loadmat("heightWeight.mat")
data = mat["heightWeightData"] # mat里面除了包含数据之外还包含很多类似于header之类的metadata信息
print('data.shape(): ', data.shape)
print('Top 10 sample data: ', data[0:10])

maleIndices = data[:, 0] == 1
males = data[maleIndices]
females = data[~maleIndices]

plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.plot(males[:, 1], males[:, 2], 'bx')
plt.plot(females[:, 1], females[:, 2], 'ro', fillstyle='none') # fillstyle='none'表明圆点是空心的
plt.xticks(np.arange(55, 90, 5))
plt.yticks(np.arange(80, 300, 20))
plt.title('red = female, blue = male')
plt.xlabel('height')
plt.ylabel('weight')

# 用高斯分布分别给male和female建模，用MLE来估计均值和协方差
# scipy.stats.multivariate_normal不提供fit方法，只好自己算
mu_male = np.average(males, axis=0)[1:]
mu_female = np.average(females, axis=0)[1:]
sigma_male = np.zeros((2, 2))

for i in range(len(males)):
    xi = males[i, 1:]
    gap = (xi - mu_male).reshape(-1, 1)
    sigma_male += np.dot(gap, gap.T)

sigma_male /= len(males)

sigma_female = np.zeros((2, 2))
for i in range(len(females)):
    xi = females[i, 1:]
    gap = (xi - mu_female).reshape(-1, 1)
    sigma_female += np.dot(gap, gap.T)

sigma_female /= len(females)

print("mu_male: ", mu_male)
print('mu_female: ', mu_female)
print('sigma_male: \n', sigma_male)
print('sigma_female: \n', sigma_female)

gaussian_male = ss.multivariate_normal(mu_male, sigma_male)
gaussian_female = ss.multivariate_normal(mu_female, sigma_female)

x, y = np.mgrid[55:85:0.1, 80:280:0.5]
pos = np.dstack((x, y))
probs_male = gaussian_male.pdf(pos)
probs_female = gaussian_female.pdf(pos)

# 画一条概率为0.05的轮廓线，囊括95%的概率范围
level_male = probs_male.min() + (probs_male.max() - probs_male.min()) * 0.05
level_female = probs_female.min() + (probs_female.max() - probs_female.min()) * 0.05

plt.subplot(122)
plt.plot(males[:, 1], males[:, 2], 'bx')
plt.plot(females[:, 1], females[:, 2], 'ro', fillstyle='none')
plt.xticks(np.arange(55, 90, 5))
plt.yticks(np.arange(80, 300, 20))
plt.title('red = female, blue = male')
plt.xlabel('height')
plt.ylabel('weight')
plt.contour(x, y , probs_male, colors='blue', levels=[level_male])
plt.contour(x, y , probs_female, colors='red', levels=[level_female])

plt.show()
