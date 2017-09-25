import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image  # pip install pillow 

'''
书中的图片clown找不到出处，这里引用sklearn中的一个例子来说明VQ
http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html\
#sphx-glr-auto-examples-cluster-plot-color-quantization-py
'''

Ks = [4, 16, 64]
china = load_sample_image("china.jpg")
china = np.array(china, dtype=np.float64) / 255 # 转化成matplotlib中常用的color格式，[0, 1]之间

w, h, d = china.shape
print('china.shape: ', china.shape)

assert d == 3   # 简易的test语句，挺好用的。这个d代表[r, g, b]
image_array = china.reshape((w * h, d)) # 转化为二维数组，便于计算

def VQ(plotindex, K):
    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]  # 随机取一千个作为训练集
    kmeans = KMeans(n_clusters=K, random_state=0).fit(image_array_sample)
    print("done in {0:.3}s.".format(time() - t0))

    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in {0:.3}s.".format(time() - t0))

    # reconstruction
    imaga_reconstruct = np.zeros((w, h, d))
    index = 0
    for i in range(w):
        for j in range(h): 
            imaga_reconstruct[i, j] = kmeans.cluster_centers_[labels[index]]
            index += 1

    plt.subplot((int)('22' + str(plotindex + 2)))
    plt.axis('off')
    plt.title('Quantized image ({0} colors, K-Means)'.format(K))
    plt.imshow(imaga_reconstruct)

fig = plt.figure(figsize=(11, 10))
plt.subplot(221)
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

for i in range(len(Ks)):
    VQ(i, Ks[i])

plt.show()
    
