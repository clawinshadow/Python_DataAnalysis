import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FuncFormatter, Formatter, FixedLocator
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['axes.axisbelow'] = False

# 其实并没有用到这个类，用了个小trick，仅仅只是修改了axis的formatter，来伪装一下
# 但这并不知正道，我觉得标准的方法还是要先transform了再画图，但是这个类始终调不通
# 以后有时间再看
class CustomXScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

    def get_transform(self):
        return self.EqualIntervalTransform()

    def set_default_locators_and_formatters(self, axis):

        class FixFormatter(Formatter):
            def __call__(self, x, pos=None):
                return x

        axis.set_major_locator(FixedLocator([1, 2, 3]))
        axis.set_major_formatter(FixFormatter())
        axis.set_minor_formatter(FixFormatter())

    class EqualIntervalTransform(mtransforms.Transform):
        input_dims = 1
        output_dims  = 1
        is_separable = True
        has_inverse = True
        
        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            print(a)
            codebook = np.array([[103, 3], [10, 2], [1, 1]])
            count = len(a.ravel())
            res = np.zeros(count)
            for i in range(count):
                ai = a.ravel()[i]
                for j in range(len(codebook)):
                    c = codebook[j]
                    if c[0] == ai:
                        res[i] = c[1]
            res = res.reshape(a.shape)
            return res
            #return np.array([1, 2, 3])

        def inverted(self):
            return CustomXScale.InvertedEqualIntervalTransform()

    class InvertedEqualIntervalTransform(mtransforms.Transform):
        input_dims = 1
        output_dims  = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            codebook = np.array([[103, 3], [10, 2], [1, 1]])
            count = len(a.ravel())
            res = np.zeros(count)
            for i in range(count):
                ai = a.ravel()[i]
                for j in range(len(codebook)):
                    c = codebook[j]
                    if c[1] == ai:
                        res[i] = c[0]
            res = res.reshape(a.shape)
            return res

        def inverted(self):
            return CustomXScale.EqualIntervalTransform()

x = np.array([103, 10, 1])
x_fake = np.array([3, 2, 1])  # a little trick
y = np.array([33, 10, 15])

plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.xlim(110, 0)
plt.plot(x, y, 'r-')

# plot with customed scale
def fakeXticks(x, pos):
    if x == 1:
        return 1
    elif x == 2:
        return 10
    elif x == 3:
        return 103
    else:
        return x

formatter = FuncFormatter(fakeXticks)

ax = plt.subplot(122)
plt.axis([3, 1, 0, 35])
plt.xticks(x_fake)
ax.xaxis.set_major_formatter(formatter)
plt.yticks(np.linspace(0, 35, 8))
plt.plot(x_fake, y, 'r-')

plt.show()
