import numpy as np
import matplotlib.pyplot as plt

'''关于离散概率分布最简单的demo'''

def Draw(ax, probs):
    ax.bar(x, probs, align='center')  # 这不是一个histogram，只能用bar来画
    plt.xticks(x)
    # 一旦指定xlim后，会自动关闭坐标轴的自动缩放, Setting limits turns autoscaling off for the x-axis.
    plt.xlim([min(x) - .5, max(x) + .5])
    plt.yticks(np.linspace(0, 1, 5))

x = np.arange(1, 5)
uni_probs = np.tile(1/4, 4)
indicator_probs = np.array([1, 0, 0, 0])

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(10, 5) 
fig.canvas.set_window_title("Discrete Probs Distribution Demo")
Draw(axs[0], uni_probs)
Draw(axs[1], indicator_probs)

plt.show()
