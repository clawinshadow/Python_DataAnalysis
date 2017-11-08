import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# laod data
data = sio.loadmat('ngramData.mat')
print(data.keys())
ugramsNorm = data['ugramsNorm']
print(ugramsNorm)

labels = '-abcdefghijklmnopqrstuvwxyz'
assert len(labels) == len(ugramsNorm)
ticklabels = ['']
for i in range(len(labels)):
    prob = np.asscalar(ugramsNorm[i])
    alphabet = labels[i]
    ticklabels.append('{0:<4}{1:.5f}{2:>4}'.format(i + 1, prob, alphabet))
ticklabels.append('')

N = len(ugramsNorm)
MAX = np.max(ugramsNorm)
MIN = np.min(ugramsNorm)
ratios = (ugramsNorm - MIN) / MAX

# plot Unigrams
fig = plt.figure(figsize=(7, 6))
fig.canvas.set_window_title('ngramPlot')
# use GridSpec to adjust subplots size
# 1. 使用2行2列来调整Bigram这个title和top xtickslabel重合的问题，gs[0, 1]专门用来画title
# 2. hspace用于调整图像的垂直间距，尽可能的相近
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 10], height_ratios=[1, 25], hspace=0)

ax = fig.add_subplot(gs[:, 0])
plt.axis([0, 1, N + 1, 0])
plt.yticks(np.linspace(0, N + 1, N + 2))
print(ax.get_position())
# ax.set_position([0.125, 0.11, 0.2, 0.88])  # [left, bottom, width, height]
ax.get_xaxis().set_visible(False)
ax.set_yticklabels(ticklabels, fontdict={'family': 'monospace'})  # 为了对齐，需要使用等宽字体
ax.tick_params(axis=u'both', which=u'both', length=0)  # hidden tick spines by setting length to 0
plt.title('Unigram')
plt.axhspan(0, N + 1, 0, 1, color='k')     # background color: black
for i in range(N):
    xmin = np.asscalar(0.5 - 0.45 * ratios[i])
    xmax = np.asscalar(0.5 + 0.45 * ratios[i])
    ymin = np.asscalar(i + 1 - 0.45 * ratios[i])
    ymax = np.asscalar(i + 1 + 0.45 * ratios[i])
    plt.axhspan(ymin, ymax, xmin, xmax, color='w') # foreground color: white

# load bigrams data
bigrams = data['bigrams']  # 这个要用未经Norm过的数据
print(bigrams)
N2 = len(bigrams)
MAX2 = np.max(bigrams)
MIN2 = np.min(bigrams)
ratios2 = (bigrams - MIN2) / MAX2
labels2 = ' -abcdefghijklmnopqrstuvwxyz '  # 29
labels2 = np.array(list(labels2))

# plot bigrams
axTitle = fig.add_subplot(gs[0, 1])  # just for title
plt.axis('off')
plt.title('Bigrams', fontdict={'verticalalignment': 'center'})

ax2 = fig.add_subplot(gs[1, 1])
plt.axis([0, N2 + 1, N2 + 1, 0])  # 29 * 29
plt.yticks(np.linspace(0, N2 + 1, N2 + 2))  # 29
plt.xticks(np.linspace(0, N2 + 1, N2 + 2))
ax2.set_yticklabels(labels2, fontdict={'family': 'monospace'})
ax2.set_xticklabels(labels2, fontdict={'family': 'monospace'})  # 为了对齐，需要使用等宽字体
ax2.tick_params(axis=u'both', which=u'both', length=0, top=True, bottom=False, \
                labeltop=True, labelbottom=False)  # hidden tick spines by setting length to 0

plt.axhspan(0, N2 + 1, 0, N2 + 1, color='k')     # background color: black
base_x = 1/29
for i in range(N2):
    for j in range(N2):
        xmin = np.asscalar(base_x * (j + 1) - base_x * ratios2[i, j] / 2)
        xmax = np.asscalar(base_x * (j + 1) + base_x * ratios2[i, j] / 2)
        ymin = np.asscalar(i + 1 - 0.45 * ratios2[i, j])
        ymax = np.asscalar(i + 1 + 0.45 * ratios2[i, j])
        plt.axhspan(ymin, ymax, xmin, xmax, color='w') # foreground color: white

plt.tight_layout()
plt.show()

