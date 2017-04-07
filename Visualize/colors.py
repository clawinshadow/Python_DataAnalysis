import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

'''
demos about matplotlib.colors:
A module for converting numbers or color arguments to RGB or RGBA
RGB and RGBA are sequences of, respectively, 3 or 4 floats in the range 0-1

Colormapping typically involves two steps: a data array is first mapped onto the range 0-1
using an instance of NOrmalize or of a subclass
then this number in the 0-1 range is mapped to a color using an instance of a subclass of Colormap

b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
'''

# all color names
# pprint.pprint(colors.cnames)
print(colors.rgb_to_hsv([0.75, 0.80, 0.90]))

# use color strings
x = np.linspace(0, 10, 30)
y1 = 1.1 * x + 1
y2 = 1.2 * x
y3 = 1.3 * x + 1
plt.plot(x, y1, color='r')
plt.plot(x, y2, color='b')
plt.plot(x, y3, color='c')

# use a string encoding a float in the 0-1 range
y4 = 1.5 * x - 1
plt.plot(x, y4, color='0.75') # gray

# use a HTML hex string
y5 = 1.6 * x + 2
plt.plot(x, y5, color='#eeefff')

# legal html names for colors are supported
y6 = 1.1 * x + 3
plt.plot(x, y6, color='burlywood')

# returns an RGBA tuple, a means alpha, 透明度
cmp = plt.get_cmap('Blues')
b1 = cmp(0.3)
b2 = cmp(0.6)
b3 = cmp(0.9)
y7 = [-4 for x in range(30)]
y8 = [-3 for x in range(30)]
y9 = [-2 for x in range(30)]
print(b1)
plt.plot(x, y7, color=b1)
plt.plot(x, y8, color=b2)
plt.plot(x, y9, color=b3)

plt.show()

