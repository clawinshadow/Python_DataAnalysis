import numpy as np
import matplotlib.pyplot as plt
'''
You can use a subset TeX markup in any matplotlib text string
by placing it inside a pair of dollar signs ($)

http://matplotlib.org/users/mathtext.html#mathtext-tutorial


'''

plt.axis([0, 10, 0, 10])
plt.title("Math Text Sample")

# plain text
plt.text(1, 9, r'alpha > beta')

# math text
plt.text(1, 8, r'$\alpha > \beta$')

# escape dollar sign \$
plt.text(1, 7, r'$\alpha \$ > \beta$')

# Subscripts and superscripts
# Use the '_' and '^' symbols:
plt.text(1, 6, r'$\alpha_i > \beta_i$')
plt.text(1, 5, r'$\sigma^i > \mu^j$')
plt.text(1, 4, r'$\sum_{i=0}^\infty x_i$')

# Fractions, binomials and stacked numbers
# Fractions, binomials and stacked numbers can be created with the
# \frac{}{}, \binom{}{} and \stackrel{}{} commands, respectively
plt.text(1, 3, r'$\frac{3}{4} \binom{3}{4} \stackrel{3}{4}$')
plt.text(1, 0, r'$\frac{5 - \frac{1}{x}}{4}$') # nested fraction

plt.text(3, 9, r'$(\frac{5 - \frac{1}{x}}{4})$')
plt.text(3, 8, r'$\left(\frac{5 - \frac{1}{x}}{4}\right)$')

# Radicals
plt.text(3, 7, r'$\sqrt{2}$')
plt.text(3, 6, r'$\sqrt[3]{x}$')
plt.text(3, 4, r'$\sqrt[3]{x}$', fontsize=20)

plt.show()
