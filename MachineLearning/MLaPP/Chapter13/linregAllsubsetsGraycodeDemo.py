import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(0)

# generate gray code from 0 to 1023
def graycode(i, length=10):
    # int -> binary -> string -> ndarray
    a = bin(i)
    b = str(a)[2:]  # remove '0b'
    c = b.rjust(length, '0')
    result = np.zeros(length)
    for i in range(len(c)):
        result[i] = int(c[i])

    return result

D = 10
maxCode = 2**10
grayCodes = np.zeros((maxCode, D)) # 1024 * 10
for i in range(maxCode):
    grayCodes[i] = graycode(i, D)

# generate data
N = 20
w = np.array([0.00, 1.67, 0.13, 0.00, 0.00, 1.19, 0.00, 0.04, 0.33, 0.00]).reshape(-1, 1)  # real w
mu = np.random.randn(D)
a = np.random.randn(D).reshape(-1, 1)
cov = np.dot(a, a.T) + 0.001 * np.eye(D)
X = ss.multivariate_normal(mu, cov).rvs(N) # N * D
sigma = 1                                  # noise sigma
y = np.dot(X, w) + sigma * np.random.randn(N, 1)
y = y - np.mean(y, axis=0)                 # remove the mean of y, not X

# Fit the spike-and-slab model, not bernoulli-gaussian model, refer to 13.2
sigmaPrior = 100   # sigma(w)
pi = 0.1           # pi_0, bernoulli parameter
score = np.zeros(maxCode)  # P(model, data) = P(model) * P(data|model)
for i in range(grayCodes.shape[0]):
    gamma = grayCodes[i] 
    cov_gamma = np.diag(sigmaPrior * gamma) # D * D
    norm0 = np.sum(gamma)  # non-zero count
    C = sigma * np.dot(X, np.dot(cov_gamma, X.T)) + sigma * np.eye(N) # N * N
    logD = ss.multivariate_normal(np.zeros(N), C).logpdf(y.ravel())
    logprior = norm0 * (np.log(pi) - np.log(1 - pi)) + D * np.log(1 - pi)
    score[i] = logD + logprior

print(score.min(), score.max())

# calculate posterior
posts = np.zeros(maxCode)
escore = np.exp(score)
posts = escore / np.sum(escore)

print(posts.min(), posts.max())
print(np.sum(posts))

# calculate inclusion marginals
weighted_models = posts.reshape(-1, 1) * grayCodes 
marg = np.sum(weighted_models, axis=0)
print(marg)

# plot the gray codes
graycodes_forplot = np.copy(grayCodes)
halflength = (int)(graycodes_forplot.shape[1] / 2)
for i in range(halflength):
    graycodes_forplot[:, i] = grayCodes[:, D - i - 1]
    graycodes_forplot[:, D - i - 1] = grayCodes[:, i]
    
fig1 = plt.figure(figsize=(12, 6)) # 弄大点，否则图像会失真
fig1.canvas.set_window_title('linregAllsubsetsGraycodeDemo_1')

plt.subplot()
plt.imshow(graycodes_forplot.T, cmap='gray', aspect='auto', extent=[1, 1024, 10, 1])

# plot P(gamma, data)
fig2 = plt.figure()
fig2.canvas.set_window_title('linregAllsubsetsGraycodeDemo_2')
plt.subplot()
plt.title('log p(model, data)')
plt.axis([-10, 1050, -140, -30])
plt.xticks(np.linspace(0, 1000, 6))
plt.yticks(np.linspace(-140, -40, 11))
plt.plot(np.linspace(1, maxCode, maxCode), score, color='midnightblue', lw=0.5)

fig3 = plt.figure()
fig3.canvas.set_window_title('linregAllsubsetsGraycodeDemo_3')
plt.subplot()
plt.title('p(model|data)')
plt.axis([-10, 1050, 0, 0.1])
plt.xticks(np.linspace(0, 1000, 6))
plt.yticks(np.linspace(0, 0.1, 11))
plt.plot(np.linspace(1, maxCode, maxCode), posts, marker='o', fillstyle='none', \
         linestyle='none', color='midnightblue', lw=0.5)
plt.vlines(np.linspace(1, maxCode, maxCode), np.zeros(maxCode), posts, colors='midnightblue', lw=0.5)

fig4 = plt.figure()
fig4.canvas.set_window_title('linregAllsubsetsGraycodeDemo_4')
plt.subplot()
plt.title('p(gamma(j)|data')
plt.axis([0, 11, 0, 1])
plt.xticks(np.linspace(0, 10, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.bar(np.linspace(0.6, 9.6, 10), marg, color='midnightblue', edgecolor='none', align='edge')

plt.show()
